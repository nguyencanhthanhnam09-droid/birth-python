import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

cap = cv2.VideoCapture(0)

def get_camera_and_eye():

    ret, frame = cap.read()
    if not ret:
        return False, None

    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    eye_closed = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            left_eye = [face_landmarks.landmark[159], face_landmarks.landmark[145]]
            right_eye = [face_landmarks.landmark[386], face_landmarks.landmark[374]]

            left_dist = abs(left_eye[0].y - left_eye[1].y)
            right_dist = abs(right_eye[0].y - right_eye[1].y)

            if left_dist < 0.03 and right_dist < 0.03:
                eye_closed = True

    return eye_closed, frame
