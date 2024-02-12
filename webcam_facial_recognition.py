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

        # Optional: Extract and save the face with landmarks every 10th frame
        if frame_counter % 10 == 0:
            top, right, bottom, left = face_recognition.face_locations(frame)[i]
            face_image = frame[top:bottom, left:right]
            cv2.imwrite(r"C:\NEW FACE RECOGNITION\OUTPUT\face_{}_frame{}.png".format(i, frame_counter), face_image)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
