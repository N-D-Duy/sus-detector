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
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD"))
TIME_TO_TRIGGER = int(os.getenv("TIME_TO_TRIGGER"))
TIME_TO_RESET = int(os.getenv("TIME_TO_RESET"))
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
        self.warning_level_2 = set(['cleaver', 'revolver', 'assault rifle', 'screwdriver'])
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
        
        self.level_1_flag = False
        self.level_2_flag = False

    def capture_frames(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        
    
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
                
            
                # if self.frame_counter % self.process_every_n_frames != 0:
                #     continue
                
        
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
                
        
                if confidence > CONFIDENCE_THRESHOLD: 
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
            print("Saving image to Firebase Storage")
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

        try:
            frame = self.frame_queue.get_nowait() 
            self.frame_queue.put(frame)
        except:
            print("Frame is None")
            frame = None

        if level_1_detected and not self.event_triggered_level_1:
            if self.level_1_detected_start is None:
                self.level_1_detected_start = current_time
            elif current_time - self.level_1_detected_start > TIME_TO_TRIGGER:
                print("Level 1 detected")
                self.event_triggered_level_1 = True
                self.level_1_flag = True
                if frame is not None:
                    image_url = self.save_to_firebase_storage(frame, 1)
                    if image_url:
                        self.level_1_image_url = image_url
                        try:
                            timestamp = datetime.datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')
                            db.reference().child('alerts/level_1').update({
                                "detected": True,
                                "timestamp": timestamp
                            })
                            self.last_firebase_update = current_time
                            print("Level 1 alert sent to Firebase")
                        except Exception as e:
                            print(f"Firebase Level 1 update error: {e}")
        elif not level_1_detected:
            self.level_1_detected_start = None
            self.event_triggered_level_1 = False
            if self.level_1_flag and (current_time - self.last_detection_time > 30.0):
                try:
                    timestamp = datetime.datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')
                    db.reference().child('alerts/level_1').update({
                        "detected": False,
                        "timestamp": timestamp
                    })
                    self.level_1_flag = False
                    print("Level 1 alert cleared in Firebase")
                except Exception as e:
                    print(f"Firebase Level 1 clear error: {e}")

        if level_2_detected and not self.event_triggered_level_2:
            if self.level_2_detected_start is None:
                self.level_2_detected_start = current_time
            elif current_time - self.level_2_detected_start > TIME_TO_TRIGGER:
                print("Level 2 detected")
                self.event_triggered_level_2 = True
                self.level_2_flag = True
                if frame is not None:
                    image_url = self.save_to_firebase_storage(frame, 2)
                    if image_url:
                        self.level_2_image_url = image_url
                        try:
                            timestamp = datetime.datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')
                            db.reference().child('alerts/level_2').update({
                                "detected": True,
                                "timestamp": timestamp
                            })
                            self.last_firebase_update = current_time
                            print("Level 2 alert sent to Firebase")
                        except Exception as e:
                            print(f"Firebase Level 2 update error: {e}")
        elif not level_2_detected:
            self.level_2_detected_start = None
            self.event_triggered_level_2 = False
            if self.level_2_flag and (current_time - self.last_detection_time > 30.0):
                try:
                    timestamp = datetime.datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')
                    db.reference().child('alerts/level_2').update({
                        "detected": False,
                        "timestamp": timestamp
                    })
                    self.level_2_flag = False
                    print("Level 2 alert cleared in Firebase")
                except Exception as e:
                    print(f"Firebase Level 2 clear error: {e}")

        if level_1_detected or level_2_detected:
            self.last_detection_time = current_time
        
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