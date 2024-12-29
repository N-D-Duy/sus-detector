import cv2
import time
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, db, storage
from dotenv import load_dotenv
import os
import datetime
import queue
import threading
import torch
import tempfile

load_dotenv()
DB_URL = os.getenv("DB_URL")
STORAGE = os.getenv("STORAGE_BUCKET")
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {'databaseURL': DB_URL})
bucket = storage.bucket(STORAGE) 

class DetectionSystem:
    def __init__(self):
    
        self.model = YOLO('yolo11l-cls.pt')
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.frame_queue = queue.Queue(maxsize=2)
        self.results_queue = queue.Queue(maxsize=2)
        self.running = True
        

        self.warning_level_1 = set(['gasmask', 'mask', 'ski_mask', 'oxygen_mask'])
        self.warning_level_2 = set(['cleaver', 'revolver', 'assault rifle'])
        self.level_1_detected_start = None
        self.level_2_detected_start = None
        self.event_triggered_level_1 = False
        self.event_triggered_level_2 = False
        self.last_detection_time = time.time()
        self.last_firebase_update = time.time()
        
        self.process_every_n_frames = 3
        self.frame_counter = 0
        self.last_fps_time = time.time()
        self.fps = 0

    def capture_frames(self):
        cap = cv2.VideoCapture(0)
        
    
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 416)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        last_time = time.time()
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_counter += 1
                current_time = time.time()
                
                if current_time - last_time >= 1.0:
                    self.fps = self.frame_counter
                    self.frame_counter = 0
                    last_time = current_time
                
            
                if self.frame_counter % self.process_every_n_frames != 0:
                    continue
                
        
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)
                
        finally:
            cap.release()

    def process_frames(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                
        
                with torch.no_grad():
                    results = self.model.predict(frame, verbose=False)
                    
        
                top_class_idx = results[0].probs.top1
                class_name = self.model.names[top_class_idx]
                confidence = float(results[0].probs.data[top_class_idx])
                
        
                if confidence > 0.3: 
                    detected_class = {class_name}
                else:
                    detected_class = set()
                
            
                self.handle_detections(detected_class)
                
                if self.results_queue.full():
                    try:
                        self.results_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.results_queue.put((frame, class_name, confidence))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")


    def save_to_firebase_storage(self, frame, level):
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg') as temp_file:
                cv2.imwrite(temp_file.name, frame)
                
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                storage_path = f"detections/level_{level}/{timestamp}.jpg"
                
                blob = bucket.blob(storage_path)
                blob.upload_from_filename(temp_file.name)
                
                image_url = blob.public_url
                
                return image_url
        except Exception as e:
            print(f"Error saving image to Firebase Storage: {e}")
            return None
    
    def handle_detections(self, detected_class):
        current_time = time.time()
        level_1_detected = bool(detected_class & self.warning_level_1)
        level_2_detected = bool(detected_class & self.warning_level_2)
        
        updates = {}
        
        if level_1_detected or level_2_detected:
            self.last_detection_time = current_time
        
        try:
            frame = self.frame_queue.queue[0]
        except:
            frame = None

        if level_1_detected and not self.event_triggered_level_1:
            if self.level_1_detected_start is None:
                self.level_1_detected_start = current_time
            elif current_time - self.level_1_detected_start > 3:
                updates['level_1'] = True
                self.event_triggered_level_1 = True
                if frame is not None:
                    image_url = self.save_to_firebase_storage(frame, 1)
                    if image_url:
                        updates['level_1_image'] = image_url
        elif not level_1_detected:
            self.level_1_detected_start = None
            self.event_triggered_level_1 = False
        
        if level_2_detected and not self.event_triggered_level_2:
            if self.level_2_detected_start is None:
                self.level_2_detected_start = current_time
            elif current_time - self.level_2_detected_start > 3:
                updates['level_2'] = True
                self.event_triggered_level_2 = True
                if frame is not None:
                    image_url = self.save_to_firebase_storage(frame, 2)
                    if image_url:
                        updates['level_2_image'] = image_url
        elif not level_2_detected:
            self.level_2_detected_start = None
            self.event_triggered_level_2 = False
        
        if current_time - self.last_detection_time > 30:
            updates['level_1'] = False
            updates['level_2'] = False
            self.last_detection_time = current_time
        
        if updates and current_time - self.last_firebase_update >= 1.0:
            try:
                timestamp = datetime.datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')
                updates_ref = {
                    "alerts": {
                        "level_1": {
                            "detected": updates.get('level_1', False),
                            "timestamp": timestamp,
                            "image_url": updates.get('level_1_image', None)
                        },
                        "level_2": {
                            "detected": updates.get('level_2', False),
                            "timestamp": timestamp,
                            "image_url": updates.get('level_2_image', None)
                        }
                    }
                }

                for level in ["level_1", "level_2"]:
                    if updates_ref["alerts"][level]["image_url"] is None:
                        del updates_ref["alerts"][level]["image_url"]

                db.reference().update(updates_ref)
                self.last_firebase_update = current_time
                print("Firebase updated successfully")
            except Exception as e:
                print(f"Firebase update error: {e}")


    def display_frames(self):
        while self.running:
            try:
                frame, class_name, confidence = self.results_queue.get(timeout=0.1)
                
                cv2.putText(frame, f"FPS: {self.fps}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if confidence > 0.3:
                    detection_text = f"{class_name}: {confidence:.2f}"
                    cv2.putText(frame, detection_text, (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
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