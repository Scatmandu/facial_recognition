import cv2
import face_recognition

video_capture = cv2.VideoCapture(0)  # 0 is usually the default camera

frame_counter = 0  # Initialize a frame counter

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Increment the frame counter
    frame_counter += 1

    # Find all the faces and face landmarks in the current frame of video
    face_landmarks_list = face_recognition.face_landmarks(frame)

    for i, face_landmarks in enumerate(face_landmarks_list):
        # Draw landmarks on the face
        for facial_feature in face_landmarks.keys():
            for point in face_landmarks[facial_feature]:
                cv2.circle(frame, point, 2, (0, 0, 255), -1)

        # Focus on the mouth region
        if 'top_lip' in face_landmarks and 'bottom_lip' in face_landmarks:
            # Combine top and bottom lip landmarks to get the full mouth region
            mouth_landmarks = face_landmarks['top_lip'] + face_landmarks['bottom_lip']

            # Calculate the bounding box for the mouth, enlarged by 10%
            min_x = min([point[0] for point in mouth_landmarks])
            max_x = max([point[0] for point in mouth_landmarks])
            min_y = min([point[1] for point in mouth_landmarks])
            max_y = max([point[1] for point in mouth_landmarks])
            
            # Enlarge the bounding box by 10%
            width = max_x - min_x
            height = max_y - min_y
            min_x = max(min_x - int(0.1 * width), 0)
            max_x = min(max_x + int(0.1 * width), frame.shape[1])
            min_y = max(min_y - int(0.1 * height), 0)
            max_y = min(max_y + int(0.1 * height), frame.shape[0])

            # Save the mouth region with landmarks every 10th frame
            if frame_counter % 10 == 0:
                mouth_region_with_dots = frame[min_y:max_y, min_x:max_x]
                if mouth_region_with_dots.size > 0:
                    cv2.imwrite(r"C:\NEW FACE RECOGNITION\OUTPUT\mouth_{}_frame{}.png".format(i, frame_counter), mouth_region_with_dots)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
