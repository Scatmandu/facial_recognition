import numpy as np
import cv2
import dlib

# Load normalized grayscale image from NumPy array (replace 'normalized_image.npy' with the actual file path)
normalized_image = np.load(r'C:\MOUTH STUFF\output_numpy_arrays\frame_30.npy')

# Convert normalized numpy array to 8-bit unsigned integer format (0 to 255)
grayscale_image = (normalized_image * 255).astype(np.uint8)

# Initialize the face detector (Dlib) and facial landmark predictor (Dlib)
detector = dlib.get_frontal_face_detector()
p = 'shape_predictor_68_face_landmarks.dat'  # Path to Dlib's pre-trained shape predictor model
predictor = dlib.shape_predictor(p)

# Detect faces in the grayscale image using Dlib
faces = detector(grayscale_image)

# Draw facial landmarks on the image
for face in faces:
    x, y, width, height = face.left(), face.top(), face.width(), face.height()

    # Predict facial landmarks
    shape = predictor(grayscale_image, face)
    shape = np.array([[point.x, point.y] for point in shape.parts()])

    # Draw green dots on the image for facial landmarks
    for (x, y) in shape:
        cv2.circle(grayscale_image, (x, y), 2, (0, 255, 0), -1)

# Save the image with facial landmarks as a PNG file
output_file = 'output_landmarks.png'
cv2.imwrite(output_file, grayscale_image)
print(f"Image with facial landmarks saved as {output_file}")
