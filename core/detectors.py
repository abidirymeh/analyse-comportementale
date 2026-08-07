import mediapipe as mp
import cv2

class HolisticDetector:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.holistic.process(rgb)