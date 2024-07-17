import cv2
import torch
from facenet_pytorch import InceptionResnetV1
from pathlib import Path
from ultralytics import YOLO

# Load FaceNet model
resnet = InceptionResnetV1(pretrained='vggface2').eval()


# Function to preprocess images
def preprocess_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb


# Function to compare embeddings
def compare_embeddings(embedding, sample_embeddings, sample_labels, threshold=0.1):
    min_distance = float('inf')
    best_match = None

    for i, sample_embedding in enumerate(sample_embeddings):
        distance = torch.nn.functional.cosine_similarity(embedding, sample_embedding, dim=1)
        if distance > threshold and distance < min_distance:
            min_distance = distance
            best_match = sample_labels[i]

    if min_distance != float('inf') and best_match is not None:
        return best_match, min_distance.item()
    else:
        return "Unknown", min_distance




# Function to recognize faces in video
def recognize_faces_in_video(video_path, yolo_weights, sample_embeddings, sample_labels):
    # Initialize YOLOv8 model
    yolo = YOLO(yolo_weights)

    # Open video file
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform face detection using YOLOv8
        face_results = yolo.predict(frame, conf=0.40)

        for infor in face_results:
            parameters = infor.boxes
            for box in parameters:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                h, w = y2 - y1, x2 - x1

                # Crop face from the frame
                face = frame[y1:y2, x1:x2]

                # Convert face to RGB (FaceNet requires RGB input)
                face_rgb = preprocess_image(face)

                # Convert to tensor and align face
                aligned = torch.tensor(face_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

                # Compute embedding vector using FaceNet
                embedding = resnet(aligned)
                # Perform face recognition based on embedding
                identity, min_distance = compare_embeddings(embedding, sample_embeddings, sample_labels)

                # Draw rectangle around the face with identity label
                if identity != "Unknown" and min_distance <= 0.1:  # Example threshold
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, identity, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Path to video file
    video_path = 0

    # Path to YOLOv8 weights file
    yolo_weights = 'yolov8n-face.pt'

    # Directory containing sample images
    sample_dir = '  /'

    # Initialize lists for embeddings and labels
    sample_embeddings = []
    sample_labels = []

    # Load and preprocess sample images
    for person_dir in Path(sample_dir).glob('*'):
        person_name = person_dir.name  # Get the name of the person from directory name
        for image_path in person_dir.glob('*'):
            image = cv2.imread(str(image_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert to tensor and align face
            aligned = torch.tensor(image_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            # Compute embedding vector using FaceNet
            embedding = resnet(aligned)

            # Append embedding and label to lists
            sample_embeddings.append(embedding)
            sample_labels.append(person_name)

    # Recognize faces in the video
    recognize_faces_in_video(video_path, yolo_weights, sample_embeddings, sample_labels)
