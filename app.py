import sqlite3
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from shapely.geometry import Point, Polygon
from ultralytics import YOLO


# ------------------------------------------------------------
# Paths and configuration
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
DB_PATH = DATA_DIR / "zone_events.db"
OUTPUT_PATH = OUTPUT_DIR / "annotated_video.mp4"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "yolo11n.pt"  # COCO model; class 0 is person.

# Default ROI is defined as percentages of the frame dimensions.
# This makes it work with different video resolutions.
DEFAULT_ZONE = (0.20, 0.20, 0.80, 0.85)


# ------------------------------------------------------------
# Database
# ------------------------------------------------------------
def initialise_database():
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS zone_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER NOT NULL,
                object_class TEXT,
                zone_name TEXT,
                entry_time TEXT,
                exit_time TEXT,
                dwell_duration_sec REAL
            )
            """
        )
        connection.commit()


def save_event(event):
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            INSERT INTO zone_events
            (track_id, object_class, zone_name, entry_time,
             exit_time, dwell_duration_sec)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            event,
        )
        connection.commit()


def read_events():
    initialise_database()
    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query(
            "SELECT * FROM zone_events ORDER BY event_id DESC",
            connection,
        )


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_NAME)


# ------------------------------------------------------------
# Video processing
# ------------------------------------------------------------
def resize_frame(frame, max_width=1280, max_height=720):
    """Resize while preserving aspect ratio; keep dimensions even for MP4."""
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    new_width = max(2, int(width * scale))
    new_height = max(2, int(height * scale))

    new_width -= new_width % 2
    new_height -= new_height % 2

    if (new_width, new_height) != (width, height):
        frame = cv2.resize(
            frame, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
    return frame


def make_zone(width, height, zone_percentages):
    x1p, y1p, x2p, y2p = zone_percentages
    points = [
        (int(x1p * width), int(y1p * height)),
        (int(x2p * width), int(y1p * height)),
        (int(x2p * width), int(y2p * height)),
        (int(x1p * width), int(y2p * height)),
    ]
    return points, Polygon(points)


def process_video(
    video_path,
    zone_name,
    frame_step,
    confidence,
    image_size,
    zone_percentages,
):
    model = load_model()
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            "OpenCV could not open this video. Try a standard MP4, MOV or AVI."
        )

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps <= 0:
        source_fps = 25.0

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    output_fps = max(source_fps / frame_step, 1.0)

    # Do not pass macroblock_size here: some installed imageio versions
    # reject it. Even-sized output frames are already created above.
    writer = None

    zone = None
    zone_points = None
    trails = defaultdict(lambda: deque(maxlen=25))
    active_visits = {}

    processed_frames = 0
    frames_with_detections = 0
    total_detections = 0
    entry_count = 0
    exit_count = 0
    unique_track_ids = set()
    frame_number = 0

    progress = st.progress(0)
    status = st.empty()
    detection_status = st.empty()

    try:
        while True:
            ok, original_frame = capture.read()
            if not ok:
                break

            frame_number += 1

            if (frame_number - 1) % frame_step != 0:
                continue

            frame = resize_frame(original_frame)
            height, width = frame.shape[:2]

            if zone is None:
                zone_points, zone = make_zone(
                    width, height, zone_percentages
                )

                # Create the writer after the final frame dimensions are known.
                writer = imageio.get_writer(
                    str(OUTPUT_PATH),
                    fps=output_fps,
                    codec="libx264",
                    quality=8,
                    pixelformat="yuv420p",
                )

            video_seconds = (frame_number - 1) / source_fps

            # Track only people (COCO class 0). A lower confidence threshold
            # helps with small/distant people, but can increase false positives.
            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                imgsz=image_size,
                conf=confidence,
                iou=0.5,
                verbose=False,
            )[0]

            processed_frames += 1
            boxes = result.boxes
            detection_count = 0

            overlay = frame.copy()
            cv2.fillPoly(
                overlay,
                [np.array(zone_points, dtype=np.int32)],
                (30, 120, 230),
            )
            frame = cv2.addWeighted(overlay, 0.12, frame, 0.88, 0)
            cv2.polylines(
                frame,
                [np.array(zone_points, dtype=np.int32)],
                True,
                (0, 165, 255),
                3,
            )
            cv2.putText(
                frame,
                zone_name,
                (zone_points[0][0], max(25, zone_points[0][1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
            )

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy().astype(int)
                class_ids = boxes.cls.cpu().numpy().astype(int)
                scores = boxes.conf.cpu().numpy()

                track_ids = None
                if boxes.id is not None:
                    track_ids = boxes.id.cpu().numpy().astype(int)

                for index, (box, class_id, score) in enumerate(
                    zip(xyxy, class_ids, scores)
                ):
                    x1, y1, x2, y2 = box.tolist()
                    label = result.names.get(int(class_id), "person")
                    detection_count += 1
                    total_detections += 1

                    # Centre point is more suitable than the bottom-centre
                    # for an overhead/aerial camera view.
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    is_inside = zone.covers(Point(center_x, center_y))

                    track_id = None
                    if track_ids is not None and index < len(track_ids):
                        track_id = int(track_ids[index])
                        unique_track_ids.add(track_id)
                        trails[track_id].append((center_x, center_y))

                    color = (0, 210, 0) if is_inside else (255, 180, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    if track_id is None:
                        display_label = f"{label} {score:.2f}"
                    else:
                        display_label = f"ID {track_id} {label} {score:.2f}"

                    cv2.putText(
                        frame,
                        display_label,
                        (x1, max(20, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

                    if track_id is not None:
                        trail_points = list(trails[track_id])
                        for trail_index in range(1, len(trail_points)):
                            cv2.line(
                                frame,
                                trail_points[trail_index - 1],
                                trail_points[trail_index],
                                (255, 255, 0),
                                2,
                            )

                        # Entry event: person transitions from outside to inside.
                        if is_inside and track_id not in active_visits:
                            active_visits[track_id] = {
                                "start_video_seconds": video_seconds,
                                "entry_time": datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                                "object_class": label,
                            }
                            entry_count += 1

                        # Exit event: tracked person transitions from inside to outside.
                        elif not is_inside and track_id in active_visits:
                            visit = active_visits.pop(track_id)
                            dwell = max(
                                0.0,
                                video_seconds - visit["start_video_seconds"],
                            )
                            save_event(
                                (
                                    track_id,
                                    visit["object_class"],
                                    zone_name,
                                    visit["entry_time"],
                                    datetime.now().strftime(
                                        "%Y-%m-%d %H:%M:%S"
                                    ),
                                    round(dwell, 2),
                                )
                            )
                            exit_count += 1

            if detection_count > 0:
                frames_with_detections += 1

            cv2.putText(
                frame,
                f"Frame: {frame_number} | People detected: {detection_count}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            # OpenCV uses BGR; imageio expects RGB.
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if total_frames > 0:
                progress.progress(min(frame_number / total_frames, 1.0))

            status.text(
                f"Processing frame {frame_number}/{total_frames or '?'} "
                f"| sampled: {processed_frames} "
                f"| detections: {total_detections} "
                f"| currently inside zone: {len(active_visits)}"
            )
            detection_status.caption(
                f"Frames with detections: {frames_with_detections} "
                f"| unique tracked IDs: {len(unique_track_ids)}"
            )

        # Close visits that are still active at the end of the clip.
        # Their exit time is the end of the clip, not a confirmed physical exit.
        final_video_seconds = max(
            0.0, (frame_number - 1) / source_fps
        )
        for track_id, visit in list(active_visits.items()):
            dwell = max(
                0.0,
                final_video_seconds - visit["start_video_seconds"],
            )
            save_event(
                (
                    int(track_id),
                    visit["object_class"],
                    zone_name,
                    visit["entry_time"],
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    round(dwell, 2),
                )
            )

        progress.progress(1.0)
        status.success(
            f"Finished: {processed_frames} sampled frames, "
            f"{total_detections} person detections, "
            f"{len(unique_track_ids)} unique tracked IDs, "
            f"{entry_count} zone entries and {exit_count} observed exits."
        )

        if total_detections == 0:
            st.warning(
                "No people were detected. Try a clearer video, increase "
                "the image size, or lower the confidence threshold."
            )
        elif len(unique_track_ids) == 0:
            st.warning(
                "People were detected, but ByteTrack did not assign track IDs. "
                "Zone visit events require tracking IDs."
            )
        elif entry_count == 0:
            st.warning(
                "People were detected, but none entered the configured zone. "
                "Adjust the zone coordinates in the sidebar so the polygon "
                "covers the area you want to monitor."
            )

    except Exception:
        raise
    finally:
        capture.release()
        if writer is not None:
            writer.close()

    if writer is None:
        raise RuntimeError("The uploaded video did not contain readable frames.")

    return (
        OUTPUT_PATH,
        processed_frames,
        total_detections,
        len(unique_track_ids),
        entry_count,
        exit_count,
    )


# ------------------------------------------------------------
# Streamlit interface
# ------------------------------------------------------------
st.set_page_config(
    page_title="Real-Time Video Analytics",
    page_icon="🎥",
    layout="wide",
)

initialise_database()

st.title("🎥 Real-Time Video Analytics")
st.caption(
    "YOLO person detection • ByteTrack multi-object tracking • "
    "Polygon geofencing • SQLite event logging"
)

st.info(
    "This app processes uploaded video files; it is not a live CCTV stream. "
    "It uses YOLO11n trained on COCO and detects the person class."
)

with st.sidebar:
    st.header("Pipeline settings")

    zone_name = st.text_input("Zone name", "Restricted Zone A")

    frame_step = st.selectbox(
        "Process every Nth frame",
        [1, 2, 3, 5],
        index=0,
        help="Use 1 for best tracking continuity. Higher values run faster "
             "but can cause missed detections and broken tracks.",
    )

    confidence = st.slider(
        "Detection confidence",
        min_value=0.05,
        max_value=0.80,
        value=0.10,
        step=0.05,
        help="Lower values can help detect small/distant people, "
             "but may add false detections.",
    )

    image_size = st.select_slider(
        "YOLO inference image size",
        options=[640, 800, 960, 1280],
        value=1280,
        help="Larger sizes can improve small-person detection but take longer.",
    )

    st.subheader("Detection zone (percentage of frame)")
    st.caption("Adjust this rectangle to cover the area you want to monitor.")

    zone_x1 = st.slider("Left edge", 0, 80, 20, 1) / 100
    zone_y1 = st.slider("Top edge", 0, 80, 20, 1) / 100
    zone_x2 = st.slider("Right edge", 20, 100, 80, 1) / 100
    zone_y2 = st.slider("Bottom edge", 20, 100, 85, 1) / 100

    if zone_x2 <= zone_x1 or zone_y2 <= zone_y1:
        st.error("The right and bottom edges must be greater than left and top.")

    st.caption(
        "Tip: For overhead footage, place the zone around the pedestrian "
        "area, not around the whole frame."
    )

uploaded_video = st.file_uploader(
    "Upload a CCTV / pedestrian video",
    type=["mp4", "mov", "avi"],
)

if uploaded_video is not None:
    safe_name = Path(uploaded_video.name).name
    input_path = DATA_DIR / safe_name
    input_path.write_bytes(uploaded_video.getbuffer())

    st.video(uploaded_video)

    if st.button("▶ Process video", type="primary"):
        if zone_x2 <= zone_x1 or zone_y2 <= zone_y1:
            st.error("Please correct the zone coordinates in the sidebar.")
        else:
            try:
                with st.spinner(
                    "Loading YOLO11n and processing your video. "
                    "The first run may download the model..."
                ):
                    (
                        output_path,
                        frame_count,
                        detections,
                        tracked_ids,
                        entries,
                        exits,
                    ) = process_video(
                        input_path,
                        zone_name,
                        frame_step,
                        confidence,
                        image_size,
                        (zone_x1, zone_y1, zone_x2, zone_y2),
                    )

                st.success("Video processing finished.")

                st.write(
                    f"**Sampled frames:** {frame_count}  |  "
                    f"**Person detections:** {detections}  |  "
                    f"**Tracked IDs:** {tracked_ids}  |  "
                    f"**Entries:** {entries}  |  **Observed exits:** {exits}"
                )

                output_bytes = output_path.read_bytes()
                st.video(output_bytes)

                st.download_button(
                    "Download annotated MP4",
                    data=output_bytes,
                    file_name="annotated_video.mp4",
                    mime="video/mp4",
                )

            except Exception as error:
                st.error("Video processing failed.")
                st.exception(error)

st.divider()
st.subheader("📊 Historical Zone Analytics")

events_df = read_events()

if events_df.empty:
    st.info(
        "No completed zone visits have been saved yet. Upload a video and "
        "process a clip where a tracked person enters or leaves the zone."
    )
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total logged visits", len(events_df))
    col2.metric("Unique tracked IDs", events_df["track_id"].nunique())
    col3.metric(
        "Average dwell time",
        f"{events_df['dwell_duration_sec'].mean():.2f} sec",
    )

    events_df["entry_hour"] = pd.to_datetime(
        events_df["entry_time"], errors="coerce"
    ).dt.hour

    hourly_events = (
        events_df.groupby("entry_hour")
        .size()
        .reset_index(name="event_count")
    )

    chart = px.bar(
        hourly_events,
        x="entry_hour",
        y="event_count",
        title="Zone visits by entry hour",
        labels={
            "entry_hour": "Entry hour",
            "event_count": "Number of visits",
        },
    )
    st.plotly_chart(chart, use_container_width=True)
    st.dataframe(events_df, use_container_width=True)

    st.download_button(
        "Export events as CSV",
        data=events_df.to_csv(index=False),
        file_name="zone_events.csv",
        mime="text/csv",
    )

st.caption(
    "Detection and tracking quality depend on camera angle, resolution, "
    "lighting, occlusion, object size and the selected zone. A visit still "
    "active at the end of a clip is closed at the clip end for reporting."
)
