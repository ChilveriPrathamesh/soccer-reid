# ⚽ Soccer Player Re-Identification using YOLOv11

This project uses a YOLOv11-based object detection model to detect and **re-identify soccer players** in a single video feed. The system ensures that:
- Players are assigned consistent IDs based on their **jersey numbers** using OCR.
- Referees are detected and labeled as `"Referee"`.
- Object tracking is done across frames to maintain identity consistency.

---

## 📁 Project Structure
soccer-reid/
├── best.pt # Trained YOLOv11 model (❌ excluded from GitHub)
├── 15sec_input_720p.mp4 # Sample input soccer video
├── detect_and_track.py # Main tracking and detection script
├── requirements.txt # Python dependencies
├── .gitignore # Excludes best.pt, venv/, etc.
└── README.md # Project documentation

---

## 🚀 Features

- ✅ Detects soccer players and referees
- ✅ Tracks players across frames using centroids
- ✅ Assigns consistent IDs using jersey number OCR
- ✅ Real-time visualization using OpenCV
- ✅ Modular code for easy training/model switching

---

## 🧰 Requirements
ultralytics==8.0.177
opencv-python
numpy
scipy


📹 How to Run
bash
Copy code
python detect_and_track.py

Output Details
🟥 Players are labeled as Player #<jersey_number>

🟦 Referees are labeled as Referee

Bounding boxes are drawn with color-coded overlays

Live OpenCV window shows the re-identification in real time
