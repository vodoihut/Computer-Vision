import cv2
from ultralytics import YOLO
import cvzone

video = r'test1.mp4'

cap = cv2.VideoCapture(video)
face_model = YOLO('yolov8n-face.pt')

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1020, 720))
    face_results = face_model.predict(frame, conf = 0.40)
    for infor in face_results:
        parameters = infor.boxes
        for box in parameters:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            h, w = y2 - y1, x2 - x1
            cvzone.cornerRect(frame, [x1,y1,w,h], l=9)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()