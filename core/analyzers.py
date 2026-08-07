import numpy as np

class AttentionAnalyzer:
    def is_looking_at_screen(self, person, img_w, img_h):
        if not person.landmarks.face_landmarks:
            return False
        face = person.landmarks.face_landmarks.landmark
        
        left_eye = np.array([face[33].x, face[33].y])
        right_eye = np.array([face[263].x, face[263].y])
        nose = np.array([face[1].x, face[1].y])
        
        eye_center = (left_eye + right_eye) / 2
        gaze_vector = nose - eye_center
        horizontal_gaze = gaze_vector[0]  
        
        return abs(horizontal_gaze) < 0.25

class PostureAnalyzer:
    def evaluate(self, person):
        if not person.landmarks.pose_landmarks:
            return "Inconnue"
        p = person.landmarks.pose_landmarks.landmark
        
        nose_y = p[0].y
        shoulder_y = (p[11].y + p[12].y) / 2
        hip_y = (p[23].y + p[24].y) / 2
        
        if nose_y > shoulder_y + 0.1:
            return "Mauvaise (tête baissée)"
        if abs(p[11].y - p[12].y) > 0.08:
            return "Mauvaise (épaules asym.)"
        return "Bonne"