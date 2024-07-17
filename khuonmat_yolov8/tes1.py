import cv2
from ultralytics import YOLO
import cvzone
import torch
from facenet_pytorch import InceptionResnetV1, extract_face

# Load FaceNet model
resnet = InceptionResnetV1(pretrained='vggface2').eval()

video = r'test1.mp4'

cap = cv2.VideoCapture(video)
face_model = YOLO('yolov8n-face.pt')

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1020, 720))
    face_results = face_model.predict(frame, conf=0.40)

    for infor in face_results:
        parameters = infor.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            h, w = y2 - y1, x2 - x1

            # Crop face from the frame
            face = frame[y1:y2, x1:x2]

            # Convert face to RGB (FaceNet requires RGB input)
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            # Convert to tensor and align face
            aligned = torch.tensor(face_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            # Compute embedding vector using FaceNet
            embedding = resnet(aligned)

            # Perform face recognition based on embedding
            # (You need to implement your own logic for face recognition)
            # Example: compare `embedding` with known embeddings
            # and decide the identity based on similarity threshold

            # Draw rectangle around the face
            cvzone.cornerRect(frame, [x1, y1, w, h], l=9)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
