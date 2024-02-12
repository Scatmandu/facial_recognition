Facial Recognition Project README

Welcome to the Facial Recognition Project! This project includes a series of Python scripts designed to facilitate various tasks related to facial detection, facial landmark extraction, image processing, and conversion between data formats. Below you will find a detailed overview of each script included in the project.

Scripts Overview
1. csv_to_numpy.py
Converts CSV files with image data back to NumPy arrays. Useful for when you have image data saved in CSV format and need to convert it back for processing with tools that require NumPy array inputs.

2. histogram_landmarks.py
Applies histogram equalization to images loaded from NumPy arrays to enhance the contrast before detecting faces and facial landmarks using Dlib. It saves the processed images with facial landmarks annotated as PNG files.

3. process_and_save_landmarks_every_fifth_frame.py
Processes every fifth frame from a directory of NumPy array-stored images, detecting faces and facial landmarks using Dlib, and saves the annotated images in a specified output directory.

4. render_from_csv.py
Reads image data from a CSV file, reconstructs the image, and saves it back as an image file. This script demonstrates how to work with image data that's been flattened and saved in CSV format.

5. save_normalized_face_regions_to_csv.py
Detects faces in video frames using MTCNN, normalizes the detected face regions, and saves these regions as CSV files. This script is helpful for preprocessing video data for further facial analysis tasks.

6. save_normalized_face_regions_to_numpy.py
Similar to save_normalized_face_regions_to_csv.py, but saves the normalized face regions as NumPy arrays. This is useful for immediate processing with Python libraries that operate on NumPy arrays.

7. save_normalized_faces_from_video.py
Detects faces in a video file using MTCNN, normalizes the face regions, and saves the regions as NumPy arrays. This script is designed for batch processing of video files to extract and normalize face data.

8. single_image_landmarks.py
Loads a single image from a NumPy array, detects faces and facial landmarks using Dlib, and saves the annotated image. This script is useful for testing or demonstration purposes with individual images.

9. webcam_facial_recognition.py
Utilizes a webcam to perform real-time facial detection and landmark annotation. This script demonstrates how to apply facial recognition technologies in a real-time context.

10. webcam_mouth_focus.py
Focuses on detecting the mouth region in real-time using a webcam, highlighting the mouth landmarks. This script is specifically designed for applications that require detailed analysis of the mouth area, such as speech analysis or emotion detection.

Dependencies
Python 3.x
NumPy
OpenCV
Dlib
MTCNN (for scripts involving MTCNN)
face_recognition (for scripts involving facial landmark detection with face_recognition)
Please ensure you have all dependencies installed to run the scripts successfully.

Installation
You can install the required libraries using pip. For example:

pip install numpy opencv-python dlib mtcnn face_recognition

Usage
Each script can be run from the command line. For example, to run csv_to_numpy.py, you can use:

python csv_to_numpy.py

Ensure you modify the script to point to the correct file paths and adjust parameters as necessary for your specific use case.