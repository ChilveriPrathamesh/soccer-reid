import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# Load your custom YOLOv11 model (ensure 'best.pt' is in the same directory)
model = YOLO("best.pt")

class PlayerTracker:
    def __init__(self, max_distance=50):
        self.next_id = 0
        self.players = {}  # player_id -> deque of previous centroids
        self.max_distance = max_distance

    def update(self, detections):
        current_ids = {}
        used_ids = set()

        for box in detections:
            x1, y1, x2, y2 = map(int, box[:4])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            matched_id = None
            for pid, history in self.players.items():
                if pid in used_ids:
                    continue
                prev_cx, prev_cy = history[-1]
                dist = np.linalg.norm(np.array([cx, cy]) - np.array([prev_cx, prev_cy]))
                if dist < self.max_distance:
                    matched_id = pid
                    self.players[pid].append((cx, cy))
                    used_ids.add(pid)
                    break

            if matched_id is None:
                matched_id = self.next_id
                self.players[matched_id] = deque([(cx, cy)], maxlen=30)
                self.next_id += 1
                used_ids.add(matched_id)

            current_ids[matched_id] = (x1, y1, x2, y2)

        return current_ids

def run_tracking(video_path):
    cap = cv2.VideoCapture(video_path)
    tracker = PlayerTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
        player_ids = tracker.update(detections)

        for pid, (x1, y1, x2, y2) in player_ids.items():
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Player {pid}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Soccer Player Re-Identification", frame)
        if cv2.waitKey(1) == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_tracking("15sec_input_720p.mp4")
