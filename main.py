import cv2
import mediapipe as mp
import numpy as np
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode = False,
    max_num_faces = 1,
    refine_landmarks = True,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)
mp_drawing = mp.solutions.drawing_utils






def eye_aspect_ratio(landmarks, eye_indices):
    top = landmarks.landmark[eye_indices[0]]
    bottom = landmarks.landmark[eye_indices[1]]
    ear = abs(top.y - bottom.y)
    return ear

def mouse_is_opened(landmarks, mouse_indices):
    top = landmarks.landmark[mouse_indices[0]]
    bottom = landmarks.landmark[mouse_indices[1]]
    left = landmarks.landmark[mouse_indices[2]]
    right = landmarks.landmark[mouse_indices[3]]
    
    height = math.sqrt(pow(abs(top.x - bottom.x), 2) + pow(abs(top.y - bottom.y), 2))
    width = math.sqrt(pow(abs(left.x - right.x), 2) + pow(abs(left.y - right.y), 2))
    if height > width/2:
        return True
    else:
        return False

cap = cv2.VideoCapture(0)

#landmark index in mediapipe
landmark_idx = {
    'Left eye' : [159, 145], # top, bottom
    'Right eye' : [386, 374],  #top, bottom
    'Mouse' : [13, 14, 61, 291] # top, bottom, left, right
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_ear = eye_aspect_ratio(face_landmarks,landmark_idx['Left eye'])
            right_ear = eye_aspect_ratio(face_landmarks,landmark_idx['Right eye'])

            blink_threshold = 0.02
            left_eye_closed = left_ear < blink_threshold
            right_eye_closed = right_ear < blink_threshold

            cv2.putText(frame,f"Left Eye: {'Closed' if left_eye_closed else 'Open'}",(30,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0) if not left_eye_closed else (0,0,255),2)
            cv2.putText(frame,f"Right Eye: {'Closed' if right_eye_closed else 'Open'}",(30,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0) if not right_eye_closed else (0,0,255),2)

            mouse_opened = mouse_is_opened(face_landmarks,landmark_idx['Mouse'])
            cv2.putText(frame,f"Mouse : {'Open' if mouse_opened else 'Closed'}",(30,90),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0) if not mouse_opened else (0,0,255),2)

            '''
            mp_drawing.draw_landmarks(
                image = frame,
                landmark_list = face_landmarks,
                connections = mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec = None,
                connection_drawing_spec = mp_drawing.DrawingSpec(color=(0,255,0))
            )
            '''
            for key in landmark_idx:
                for idx in landmark_idx[key]:
                    point = face_landmarks.landmark[idx]
                    x = int(point.x * width)
                    y = int(point.y * height)
                    cv2.circle(frame,(x,y),5,(0,0,255),-1)

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()