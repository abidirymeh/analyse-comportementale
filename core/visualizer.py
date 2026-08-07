import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

class Visualizer:
    def draw_all(self, image, person, looking_at_screen, posture_status):
        results = person.landmarks

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)

        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image, results.face_landmarks, mp.solutions.holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec)

        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)

        color = (0, 255, 0) if looking_at_screen else (0, 0, 255)
        cv2.putText(image, "ATTENTIF" if looking_at_screen else "DISTRAIT",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        cv2.putText(image, posture_status, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (0,255,0) if "Bonne" in posture_status else (0,0,255), 3)

        cv2.putText(image, f"ID: {person.id}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        return image