import cv2
import time
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, db
from dotenv import load_dotenv
import os
import datetime

load_dotenv()
DB_URL = os.getenv("DB_URL")
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': DB_URL
})

model = YOLO('yolo11s-cls.pt')

cap = cv2.VideoCapture(0)

level_1_detected_start = None
level_2_detected_start = None
event_triggered_level_1 = False
event_triggered_level_2 = False
warning_level_1 = ['gasmask', 'mask', 'ski_mask', 'oxygen_mask']
warning_level_2 = ['cleaver', 'revolver', 'assault rifle']

last_analysis_time = 0
last_detection_time = time.time()
analysis_interval = 0.5 
check_interval = 30

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    if current_time - last_analysis_time > analysis_interval:
        last_analysis_time = current_time

        results = model.predict(frame)
        top_classes = results[0].probs.top5
        class_names = [model.names[cls_idx] for cls_idx in top_classes]

        level_1_detected = any(cls in warning_level_1 for cls in class_names)
        level_2_detected = any(cls in warning_level_2 for cls in class_names)
        
        if level_1_detected or level_2_detected:
            last_detection_time = time.time()

        if level_1_detected:
            if level_1_detected_start is None:
                level_1_detected_start = time.time()
            elif time.time() - level_1_detected_start > 3 and not event_triggered_level_1:
                print("Triggering Level 1 event: Updating Firebase")
                try:
                    timestamp = time.time()
                    formatted_timestamp = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                    db.reference('alerts/level_1').set({"detected": True, "timestamp": formatted_timestamp})
                    print("Firebase Level 1 updated successfully")
                except Exception as e:
                    print(f"Firebase Level 1 update error: {e}")
                event_triggered_level_1 = True
        else:
            event_triggered_level_1 = False
            level_1_detected_start = None

        if level_2_detected:
            if level_2_detected_start is None:
                level_2_detected_start = time.time()
            elif time.time() - level_2_detected_start > 3 and not event_triggered_level_2:
                print("Triggering Level 2 event: Updating Firebase")
                try:
                    timestamp = time.time()
                    formatted_timestamp = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                    db.reference('alerts/level_2').set({"detected": True, "timestamp": formatted_timestamp})
                    print("Firebase Level 2 updated successfully")
                except Exception as e:
                    print(f"Firebase Level 2 update error: {e}")
                event_triggered_level_2 = True
        elif not level_2_detected:
            event_triggered_level_2 = False
            level_2_detected_start = None

    if time.time() - last_detection_time > check_interval:
        try:
            imestamp = time.time() 
            formatted_timestamp = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            db.reference('alerts/level_1').set({"detected": False, "timestamp": formatted_timestamp})
            db.reference('alerts/level_2').set({"detected": False, "timestamp": formatted_timestamp})
            print("No sus detection for 30 seconds, Firebase reset to False")
        except Exception as e:
            print(f"Firebase reset error: {e}")
        last_detection_time = time.time()
        
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
