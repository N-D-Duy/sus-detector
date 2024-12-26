import cv2
import time
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, db
from dotenv import load_dotenv
import os
import datetime
import queue
import threading
from collections import deque
import numpy as np

load_dotenv()
DB_URL = os.getenv("DB_URL")
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {'databaseURL': DB_URL})

class DetectionSystem:
    def __init__(self):
        self.model = YOLO('yolo11s-cls.pt')
        self.frame_queue = queue.Queue(maxsize=4) 
        self.results_queue = queue.Queue()
        self.running = True
        
        self.warning_level_1 = set(['gasmask', 'mask', 'ski_mask', 'oxygen_mask'])
        self.warning_level_2 = set(['cleaver', 'revolver', 'assault rifle'])
        self.level_1_detected_start = None
        self.level_2_detected_start = None
        self.event_triggered_level_1 = False
        self.event_triggered_level_2 = False
        self.last_detection_time = time.time()
        self.last_firebase_update = time.time()
        self.firebase_update_interval = 1.0
        
        self.fps_buffer = deque(maxlen=30)
        self.last_fps_time = time.time()
        self.frame_count = 0

    def capture_frames(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if self.frame_queue.full():
                    continue
                    
                self.frame_queue.put(frame)
                
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    current_time = time.time()
                    fps = 30 / (current_time - self.last_fps_time)
                    self.fps_buffer.append(fps)
                    self.last_fps_time = current_time
        finally:
            cap.release()

    def process_frames(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                
                results = self.model.predict(frame)
                top_classes = results[0].probs.top5
                class_names = set(self.model.names[cls_idx] for cls_idx in top_classes)
                
                self.handle_detections(class_names)
                
                self.results_queue.put((frame, class_names))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")

    def handle_detections(self, class_names):
        current_time = time.time()
        level_1_detected = bool(class_names & self.warning_level_1)
        level_2_detected = bool(class_names & self.warning_level_2)
        
        if level_1_detected:
            if self.level_1_detected_start is None:
                self.level_1_detected_start = current_time
            elif current_time - self.level_1_detected_start > 3 and not self.event_triggered_level_1:
                self.update_firebase("level_1", True)
                self.event_triggered_level_1 = True
        else:
            self.event_triggered_level_1 = False
            self.level_1_detected_start = None

        if level_2_detected:
            if self.level_2_detected_start is None:
                self.level_2_detected_start = current_time
            elif current_time - self.level_2_detected_start > 3 and not self.event_triggered_level_2:
                self.update_firebase("level_2", True)
                self.event_triggered_level_2 = True
        else:
            self.event_triggered_level_2 = False
            self.level_2_detected_start = None

        if current_time - self.last_detection_time > 30:
            self.update_firebase("level_1", False)
            self.update_firebase("level_2", False)
            self.last_detection_time = current_time

    def update_firebase(self, level, detected):
        current_time = time.time()
        
        if current_time - self.last_firebase_update < self.firebase_update_interval:
            return
            
        try:
            timestamp = datetime.datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')
            db.reference(f'alerts/{level}').set({
                "detected": detected,
                "timestamp": timestamp
            })
            self.last_firebase_update = current_time
        except Exception as e:
            print(f"Firebase update error: {e}")

    def display_frames(self):
        while self.running:
            try:
                frame, detected_classes = self.results_queue.get(timeout=1.0)
                
                if self.fps_buffer:
                    fps_text = f"FPS: {np.mean(self.fps_buffer):.1f}"
                    cv2.putText(frame, fps_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                y_pos = 60
                for cls in detected_classes:
                    cv2.putText(frame, cls, (10, y_pos), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y_pos += 20
                
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    
            except queue.Empty:
                continue

    def run(self):
        threads = [
            threading.Thread(target=self.capture_frames),
            threading.Thread(target=self.process_frames),
            threading.Thread(target=self.display_frames)
        ]
        
        for t in threads:
            t.daemon = True
            t.start()
            
        try:
            while self.running:
                time.sleep(0.1) 
        except KeyboardInterrupt:
            self.running = False
            
        for t in threads:
            t.join()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = DetectionSystem()
    system.run()