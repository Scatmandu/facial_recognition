import cv2
import os
from mtcnn.mtcnn import MTCNN
import numpy as np

# Initialize the face detector (MTCNN)
detector = MTCNN()

video_path = r'C:\Users\TYLER\Downloads\mixkit-four-scientists-talking-about-research-22999-medium.mp4'
output_folder = 'output_numpy_arrays'
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

    # Loop over the detected faces
    for face in faces:
        x, y, width, height = face['box']
        confidence = face['confidence']

        if confidence > 0.6:
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

            # Save the normalized frame as a numpy array
            np.save(os.path.join(output_folder, f'frame_{frame_count}.npy'), normalized_gray_frame)
            print(f"Saved frame {frame_count} as a numpy array.")  # Progress message

    frame_count += 1

cap.release()
