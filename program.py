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
import numpy as np
import face_recognition

load_dotenv()
DB_URL = os.getenv("DB_URL")
STORAGE = os.getenv("STORAGE_BUCKET")
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD"))
TIME_TO_TRIGGER = int(os.getenv("TIME_TO_TRIGGER"))
TIME_TO_RESET = int(os.getenv("TIME_TO_RESET"))
FACE_RECOGNITION_THRESHOLD = float(os.getenv("FACE_RECOGNITION_THRESHOLD", 0.6))
DOOR_OPEN_DURATION = int(os.getenv("DOOR_OPEN_DURATION", 5))

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {'databaseURL': DB_URL})
bucket = storage.bucket(STORAGE)

class DetectionSystem:
    def __init__(self):
        self.model = YOLO('yolo11l-cls.pt')
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.frame_queue = queue.Queue(maxsize=2)        
        self.face_frame_queue = queue.Queue(maxsize=2)   
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
        
        
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()
        self.face_detection_enabled = True
        self.last_face_check_time = time.time()
        self.door_open_time = None
        self.door_is_open = False
        
        
        self.door_opened_by_face_recognition = False
        
        
        self.door_lock = threading.Lock()
        
        
        try:
            door_status = db.reference().child('iot/door').get()
            self.door_is_open = bool(door_status) if door_status is not None else False
            print(f"Trạng thái cửa ban đầu: {'Mở' if self.door_is_open else 'Đóng'}")
            
            
            if self.door_is_open:
                self.door_opened_by_face_recognition = False
        except Exception as e:
            print(f"Không thể đọc trạng thái cửa từ Firebase: {e}")

    def load_known_faces(self):
        """Tải danh sách khuôn mặt đã biết từ thư mục"""
        faces_dir = "known_faces"
        if not os.path.exists(faces_dir):
            os.makedirs(faces_dir)
            print(f"Tạo thư mục {faces_dir}. Vui lòng thêm ảnh khuôn mặt vào thư mục này.")
            return
        
        print("Đang tải dữ liệu khuôn mặt đã biết...")
        for filename in os.listdir(faces_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                try:
                    image_path = os.path.join(faces_dir, filename)
                    face_image = face_recognition.load_image_file(image_path)
                    
                    face_encodings = face_recognition.face_encodings(face_image)
                    if len(face_encodings) > 0:
                        face_encoding = face_encodings[0]
                        self.known_face_encodings.append(face_encoding)
                        
                        name = os.path.splitext(filename)[0]
                        self.known_face_names.append(name)
                        print(f"Đã tải khuôn mặt: {name}")
                    else:
                        print(f"Không tìm thấy khuôn mặt trong ảnh: {filename}")
                except Exception as e:
                    print(f"Lỗi khi xử lý ảnh {filename}: {e}")
        
        print(f"Đã tải {len(self.known_face_encodings)} khuôn mặt")

    def capture_frames(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 416)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        last_time = time.time()
        face_frame_counter = 0
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("Không thể đọc frame từ camera")
                    time.sleep(0.1)
                    continue
                    
                self.frame_counter += 1
                face_frame_counter += 1
                current_time = time.time()
                
                if current_time - last_time >= 1.0:
                    self.fps = self.frame_counter
                    self.frame_counter = 0
                    last_time = current_time
                
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())
                else:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame.copy())
                    except queue.Empty:
                        pass
                        
                if face_frame_counter % 5 == 0 and not self.face_frame_queue.full():
                    self.face_frame_queue.put(frame.copy())
                    face_frame_counter = 0
                
        finally:
            cap.release()

    def process_yolo(self):
        """Xử lý frames với model YOLO"""
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
                
                self.handle_detections(detected_class, frame)
                
                if not self.results_queue.full():
                    self.results_queue.put((frame, class_name, confidence))
                else:
                    try:
                        self.results_queue.get_nowait()
                        self.results_queue.put((frame, class_name, confidence))
                    except queue.Empty:
                        pass
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"YOLO processing error: {e}")
    
    def process_face_recognition(self):
        """Xử lý nhận diện khuôn mặt trong một thread riêng"""
        while self.running:
            try:
                if not self.face_detection_enabled or len(self.known_face_encodings) == 0:
                    time.sleep(0.5)
                    continue
                    
                frame = self.face_frame_queue.get(timeout=0.5)
                self.recognize_faces(frame)
                
                
                current_time = time.time()
                with self.door_lock:
                    if (self.door_is_open and 
                        self.door_opened_by_face_recognition and 
                        self.door_open_time and 
                        current_time - self.door_open_time >= DOOR_OPEN_DURATION):
                        print(f"Đã hết thời gian mở cửa ({DOOR_OPEN_DURATION}s)")
                        self.close_door()
                
            except queue.Empty:
                
                current_time = time.time()
                with self.door_lock:
                    if (self.door_is_open and 
                        self.door_opened_by_face_recognition and 
                        self.door_open_time and 
                        current_time - self.door_open_time >= DOOR_OPEN_DURATION):
                        print(f"Đã hết thời gian mở cửa ({DOOR_OPEN_DURATION}s)")
                        self.close_door()
                continue
            except Exception as e:
                print(f"Face recognition error: {e}")
                time.sleep(0.1)  

    def recognize_faces(self, frame):
        """Nhận diện khuôn mặt trong frame"""
        if len(self.known_face_encodings) == 0:
            return
        
        try:
            
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            
            
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            
            face_locations = face_recognition.face_locations(rgb_small_frame)
            
            if len(face_locations) > 0:
                
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=FACE_RECOGNITION_THRESHOLD)
                    
                    
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                            similarity = 1 - face_distances[best_match_index]
                            print(f"Nhận diện thành công: {name} (Độ tương đồng: {similarity:.2f})")
                            if(similarity >= FACE_RECOGNITION_THRESHOLD):
                                self.open_door(name)
                            return
        except Exception as e:
            print(f"Lỗi trong quá trình nhận diện khuôn mặt: {e}")

    def open_door(self, name):
        """Mở cửa khi nhận diện thành công khuôn mặt"""
        with self.door_lock:
            if not self.door_is_open:
                print(f"Mở cửa cho: {name}")
                self.door_is_open = True
                self.door_open_time = time.time()
                self.door_opened_by_face_recognition = True  
                
                
                try:
                    db.reference().child('iot/door').set(True)
                    print("Đã gửi lệnh mở cửa đến Firebase: iot/door = true")
                except Exception as e:
                    print(f"Firebase door update error: {e}")
                    
                    
                    self.door_is_open = False
                    self.door_open_time = None
                    self.door_opened_by_face_recognition = False

    def close_door(self):
        """Đóng cửa sau khi hết thời gian"""
        with self.door_lock:
            
            if self.door_is_open and self.door_opened_by_face_recognition:
                print("Đóng cửa tự động")
                self.door_is_open = False
                self.door_open_time = None
                self.door_opened_by_face_recognition = False
                
                
                try:
                    db.reference().child('iot/door').set(False)
                    print("Đã gửi lệnh đóng cửa đến Firebase: iot/door = false")
                except Exception as e:
                    print(f"Firebase door update error: {e}")
                    
                    
                    self.door_is_open = True
                    self.door_open_time = time.time()
                    self.door_opened_by_face_recognition = True

    def firebase_listener(self):
        """Theo dõi thay đổi trạng thái cửa từ Firebase"""
        print("Bắt đầu lắng nghe thay đổi trạng thái cửa từ Firebase")
        
        def door_callback(event):
            try:
                
                if event.data is not None:
                    new_door_status = bool(event.data)
                    with self.door_lock:
                        if self.door_is_open != new_door_status:
                            old_status = self.door_is_open
                            self.door_is_open = new_door_status
                            print(f"Trạng thái cửa đã được thay đổi từ app: {old_status} → {new_door_status}")
                            
                            
                            if self.door_is_open:
                                self.door_opened_by_face_recognition = False  
                                self.door_open_time = None  
                            
                            
                            else:
                                self.door_open_time = None
                                self.door_opened_by_face_recognition = False
            except Exception as e:
                print(f"Lỗi khi xử lý thay đổi từ Firebase: {e}")

        
        door_ref = db.reference().child('iot/door')
        door_ref.listen(door_callback)
        
        
        while self.running:
            try:
                time.sleep(30)  
                with self.door_lock:
                    
                    firebase_door_state = db.reference().child('iot/door').get()
                    if firebase_door_state is not None and bool(firebase_door_state) != self.door_is_open:
                        print(f"Door state sync error detected! Firebase: {firebase_door_state}, Local: {self.door_is_open}")
                        
                        
                        db.reference().child('iot/door').set(self.door_is_open)
                        print(f"Resynced door state to Firebase: {self.door_is_open}")
            except Exception as e:
                print(f"Firebase periodic sync error: {e}")
        
        print("Đã dừng lắng nghe thay đổi từ Firebase")

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
    
    def handle_detections(self, detected_class, frame):
        current_time = time.time()
        level_1_detected = bool(detected_class & self.warning_level_1)
        level_2_detected = bool(detected_class & self.warning_level_2)
        
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
            if self.level_2_flag and (current_time - self.last_detection_time > 15.0):
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
                
                door_status = "OPEN" if self.door_is_open else "CLOSED"
                door_color = (0, 255, 0) if self.door_is_open else (0, 0, 255)
                
                
                if self.door_is_open:
                    opened_by = "Face Recog." if self.door_opened_by_face_recognition else "App"
                    auto_close = "Auto-close ON" if self.door_opened_by_face_recognition else "No auto-close"
                    
                    cv2.putText(frame, f"Door: {door_status} by {opened_by}", (10, 90),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, door_color, 1)
                    cv2.putText(frame, auto_close, (10, 110),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, door_color, 1)
                    
                    
                    if self.door_opened_by_face_recognition and self.door_open_time:
                        time_left = max(0, DOOR_OPEN_DURATION - (time.time() - self.door_open_time))
                        cv2.putText(frame, f"Auto-close in: {time_left:.1f}s", (10, 130),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, door_color, 1)
                else:
                    cv2.putText(frame, f"Door: {door_status}", (10, 90),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, door_color, 1)
                
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Display error: {e}")

    def run(self):
        threads = [
            threading.Thread(target=self.capture_frames, name="CaptureThread"),
            threading.Thread(target=self.process_yolo, name="YOLOThread"),
            threading.Thread(target=self.process_face_recognition, name="FaceRecognitionThread"),
            threading.Thread(target=self.display_frames, name="DisplayThread"),
            threading.Thread(target=self.firebase_listener, name="FirebaseListenerThread")
        ]
        
        for t in threads:
            t.daemon = True
            t.start()
            print(f"Đã khởi động thread: {t.name}")
        
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Nhận tín hiệu thoát, đang dừng các thread...")
            self.running = False
        
        for t in threads:
            t.join()
        cv2.destroyAllWindows()
        print("Chương trình đã kết thúc")

if __name__ == "__main__":
    system = DetectionSystem()
    system.run()