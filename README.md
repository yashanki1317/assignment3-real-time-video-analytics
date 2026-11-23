# Assignment 3 — Real-Time Video Analytics & Multi-Object Tracking

## Features
- YOLO object detection with persistent ByteTrack IDs.
- Object trajectory trails.
- Polygon ROI geofencing using Shapely.
- Zone entry/exit detection and dwell-time calculation.
- Background-thread SQLite logging (`zone_events`).
- Streamlit dashboard, Plotly hourly chart, event table and CSV export.
- Frame sampling and 640×360 resizing to reduce CPU load.

## Setup (macOS)
```bash
cd assignment3-real-time-video-analytics
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p models data output screenshots
```
Copy your previous model to `models/best.pt`. Keep model weights local; `.gitignore` excludes them.

## Run
```bash
streamlit run app.py
```
Upload a short MP4 and click **Process video**.

## Pipeline
1. OpenCV reads and resizes frames.
2. Frames are sampled to reduce CPU usage.
3. Ultralytics `model.track(..., tracker="bytetrack.yaml", persist=True)` detects and tracks.
4. Track IDs and trajectory points are maintained across processed frames.
5. Shapely checks the bottom-centre of each bounding box against a polygon ROI.
6. Entry/exit transitions create completed zone events; dwell time is calculated from video timestamps.
7. A background worker thread writes events into SQLite.
8. Streamlit displays video output and historical analytics.

## SQLite schema
`zone_events(event_id, track_id, object_class, zone_name, entry_time, exit_time, dwell_duration_sec)`

## Important notes
- The ROI is a sample polygon in the centre of the resized frame. Change `polygon_pts` in `app.py` to fit your camera view.
- This MVP processes uploaded video files, not a live CCTV stream.
- The app saves active zone visits at end-of-video.
- The `best.pt` model only recognizes classes it was trained on.
- IDs may switch after long occlusions; frame skipping reduces temporal precision.
- For GitHub, do not commit `.venv`, model weights, uploaded videos, SQLite DB or generated output videos unless requested.
