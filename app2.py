import cv2
import tensorflow as tf
import numpy as np
import imutils
import pickle
import os

model_path = 'liveness.model'
le_path = 'label_encoder.pickle'
encodings = 'encoded_faces.pickle'
detector_folder = 'face_detector'
confidence = 0.5

args = {'model': model_path, 'le': le_path, 'detector': detector_folder,
        'encodings': encodings, 'confidence': confidence}

# Load the encoded faces and names
print('[INFO] Loading encodings...')
with open(args['encodings'], 'rb') as file:
    encoded_data = pickle.load(file)

# Load the face detector model
print('[INFO] Loading face detector...')
proto_path = os.path.sep.join([args['detector'], 'deploy.prototxt'])
model_path = os.path.sep.join([args['detector'], 'res10_300x300_ssd_iter_140000.caffemodel'])
detector_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# Load the liveness detection model and label encoder
liveness_model = tf.keras.models.load_model(args['model'])
le = pickle.load(open(args['le'], 'rb'))

# Initialize the video capture from webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    ret, frame = video_capture.read()

    if not ret:
        break

    # Resize the frame to have a maximum width of 800 pixels
    frame = imutils.resize(frame, width=800)

    # Get the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the face detector network and obtain detections
    detector_net.setInput(blob)
    detections = detector_net.forward()

    # Iterate over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > args['confidence']:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            startX = max(0, startX - 20)
            startY = max(0, startY - 20)
            endX = min(w, endX + 20)
            endY = min(h, endY + 20)

            face = frame[startY:endY, startX:endX]
            try:
                face = cv2.resize(face, (32, 32))
            except:
                break

            name = 'Unknown'
            face = face.astype('float') / 255.0
            face = tf.keras.preprocessing.image.img_to_array(face)
            face = np.expand_dims(face, axis=0)

            preds = liveness_model.predict(face)[0]
            j = np.argmax(preds)
            label_name = le.classes_[j]

            label = f'{label_name}: {preds[j]:.4f}'
            print(f'[INFO] {name}, {label_name}')

            if label_name == 'fake':
                cv2.putText(frame, "Fake Alert!", (startX, endY + 25),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(frame, name, (startX, startY - 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 130, 255), 2)
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 4)

    # Display the resulting frame
    cv2.imshow("Liveness Detection", frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()
