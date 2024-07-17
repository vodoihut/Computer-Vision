import cv2
from ultralytics import YOLO
import cvzone
import facenet
import tensorflow as tf


VIDEO_PATH = 'test1.mp4'

face_model = YOLO('yolov8n-face.pt')
with tf.Graph().as_default():
    # Cai dat GPU neu co
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

    with sess.as_default():
        # Load model MTCNN phat hien khuon mat
        print('Loading feature extraction model')
        facenet.load_model(FACENET_MODEL_PATH)

        # Lay tensor input va output
        images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        people_detected = set()
        person_detected = collections.Counter()

        # Lay hinh anh tu file video
        cap = cv2.VideoCapture(VIDEO_PATH)

        while (cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1020, 720))

            # Phát hiện khuôn mặt bằng YOLO
            face_results = face_model.predict(frame, conf=0.40)

            # Lặp qua các khuôn mặt đã phát hiện
            for result in face_results:
                x1, y1, x2, y2 = result.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                h, w = y2 - y1, x2 - x1

                # Vẽ khung màu xanh quanh khuôn mặt
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Cắt và chuẩn bị ảnh cho FaceNet
                cropped_face = frame[y1:y2, x1:x2, :]
                scaled = cv2.resize(cropped_face, (160, 160), interpolation=cv2.INTER_CUBIC)
                scaled = facenet.prewhiten(scaled)
                scaled_reshape = scaled.reshape(-1, 160, 160, 3)

                # Đưa vào FaceNet để nhận diện
                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                emb_array = sess.run(embeddings, feed_dict=feed_dict)

                # Dự đoán bằng model classifier
                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                # Lấy ra tên và xác suất nhận dạng cao nhất
                best_name = class_names[best_class_indices[0]]

                # Hiển thị tên lên frame
                text_x = x1
                text_y = y2 + 20
                if best_class_probabilities > 0.5:
                    name = best_name
                else:
                    name = "Unknown"

                cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1, (255, 255, 255), thickness=1, lineType=2)
                cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1, (255, 255, 255), thickness=1, lineType=2)

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
