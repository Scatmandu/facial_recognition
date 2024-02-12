import cv2
import os
import dlib
from mtcnn.mtcnn import MTCNN
import numpy as np

# Initialize the face detector (MTCNN) and facial landmark predictor (Dlib)
detector = MTCNN()
p = r'C:\ml_project\shape_predictor_68_face_landmarks.dat'
dlib_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

video_path = r'C:\Users\TYLER\Downloads\mixkit-four-scientists-talking-about-research-22999-medium.mp4'
output_folder = 'output_csv_arrays'
os.makedirs(output_folder, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect faces in the frame using MTCNN
    faces = detector.detect_faces(frame)

    # Initialize a flag to check if any face was detected
    face_detected = False

    # Loop over the detected faces
    for face in faces:
        x, y, width, height = face['box']
        confidence = face['confidence']

        if confidence > 0.6:
            face_detected = True  # Set the flag to True

            # Calculate the margin to be added to the bounding box
            x_margin = int(width * 0.2)
            y_margin = int(height * 0.2)

            # Adjust the coordinates and dimensions of the bounding box
            x -= x_margin
            y -= y_margin
            width += 2 * x_margin
            height += 2 * y_margin

            # Convert the frame to grayscale for facial landmark detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Normalize the grayscale frame to range [0, 1] within the rectangle
            normalized_gray_frame = gray_frame[y:y+height, x:x+width] / 255.0

            # Save the normalized frame as a CSV file
            np.savetxt(os.path.join(output_folder, f'frame_{frame_count}.csv'), normalized_gray_frame, delimiter=',')
            print(f"Saved frame {frame_count} as a CSV file.")  # Progress message

    frame_count += 1

cap.release()
