import streamlit as st
import cv2
import numpy as np
from sympy import fps
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import time
from deepface import DeepFace
st.set_page_config(
    page_title="Analyse Comportementale",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stDataFrame {
        border: 2px solid #1E88E5;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


POSE_CONNECTIONS = [
    (0, 1), (0, 2), 
    (1, 3), (2, 4),  
    (0, 5), (0, 6),  
    
    (5, 6),  
    (5, 11), (6, 12), 
    (11, 12),
    
    (5, 7), (7, 9),
    
    (6, 8), (8, 10), 
    
    (11, 13), (13, 15), 

    (12, 14), (14, 16), 
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

def is_valid_human_pose(kpts_clean, frame_shape):
    left_shoulder = safe_kpt(kpts_clean, 5)
    right_shoulder = safe_kpt(kpts_clean, 6)
    left_hip = safe_kpt(kpts_clean, 11)
    right_hip = safe_kpt(kpts_clean, 12)
    
    if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
        return False

    valid_count = sum(1 for pt in kpts_clean if pt is not None)
    return valid_count >= 6  # ↓ de 9 à 6
def draw_pose_keypoints(frame, keypoints, track_id, color=(0, 255, 255), thickness=2):
    """
    Dessine les keypoints du corps et les connexions squelettiques
    """
    kpts_clean = [
        (int(pt[0]), int(pt[1])) if pt[0] > 0 and pt[1] > 0 else None 
        for pt in keypoints
    ]
    
    for connection in POSE_CONNECTIONS:
        pt1_idx, pt2_idx = connection
        if pt1_idx < len(kpts_clean) and pt2_idx < len(kpts_clean):
            pt1 = kpts_clean[pt1_idx]
            pt2 = kpts_clean[pt2_idx]
            
            if pt1 is not None and pt2 is not None:
                cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)
    
    for idx, pt in enumerate(kpts_clean):
        if pt is not None:
            if idx == 0: 
                point_color = (255, 0, 255)  
                radius = int(9* 2.0)   
            elif idx in [1, 2]: 
                point_color = (255, 255, 0)  
                radius = int(8*2.0)
            elif idx in [5, 6]:  
                point_color = (0, 255, 0) 
                radius = int(10*2.0)
            elif idx in [11, 12]: 
                point_color = (255, 0, 0)  
                radius = int(10*2.0)
            elif idx in [9, 10]: 
                point_color = (0, 165, 255) 
                radius = int(9*2.0)
            else:  
                point_color = color
                radius = int(8*2.0)
            
            cv2.circle(frame, pt, radius, point_color, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, radius + 2, (0, 0, 0), 1, cv2.LINE_AA)
            
           





def safe_kpt(kpts, idx):
    """Extraction sécurisée des keypoints"""
    if idx < len(kpts) and kpts[idx] is not None:
        x, y = kpts[idx]
        if x > 0 and y > 0:
            return (x, y)
    return None



class PostureFuzzyClassifier:
    def __init__(self):
        import skfuzzy as fuzz
        from skfuzzy import control as ctrl
        
        self.system_full = self._build_system(
            use_hip_ratio=True, use_trunk_angle=True, use_hip_knee_dist=True
        )
        self.system_partial = self._build_system(
            use_hip_ratio=True, use_trunk_angle=True, use_hip_knee_dist=False
        )
        self.system_minimal = self._build_system(
            use_hip_ratio=False, use_trunk_angle=True, use_hip_knee_dist=False
        )

    def _build_system(self, use_hip_ratio, use_trunk_angle, use_hip_knee_dist):
        import skfuzzy as fuzz
        from skfuzzy import control as ctrl
        
        inputs = {}
        antecedents = []
        
        if use_hip_ratio:
            hip_ratio = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'hip_ratio')
            hip_ratio['haut'] = fuzz.trimf(hip_ratio.universe, [0, 0, 0.50])
            hip_ratio['milieu'] = fuzz.trimf(hip_ratio.universe, [0.45, 0.60, 0.75])
            hip_ratio['bas'] = fuzz.trimf(hip_ratio.universe, [0.70, 1.0, 1.0])
            inputs['hip_ratio'] = hip_ratio
            antecedents.append(hip_ratio)
        
        if use_trunk_angle:
            trunk_angle = ctrl.Antecedent(np.arange(-90, 91, 1), 'trunk_angle')
            trunk_angle['arriere'] = fuzz.trimf(trunk_angle.universe, [-90, -90, -30])
            trunk_angle['vertical'] = fuzz.trimf(trunk_angle.universe, [-25, 0, 25])
            trunk_angle['avant'] = fuzz.trimf(trunk_angle.universe, [30, 90, 90])
            inputs['trunk_angle'] = trunk_angle
            antecedents.append(trunk_angle)
        
        if use_hip_knee_dist:
            hip_knee_dist = ctrl.Antecedent(np.arange(0, 201, 1), 'hip_knee_dist')
            hip_knee_dist['petite'] = fuzz.trimf(hip_knee_dist.universe, [0, 0, 70])
            hip_knee_dist['grande'] = fuzz.trimf(hip_knee_dist.universe, [130, 200, 200])
            inputs['hip_knee_dist'] = hip_knee_dist
            antecedents.append(hip_knee_dist)
        
        debout = ctrl.Consequent(np.arange(0, 101, 1), 'debout')
        assis = ctrl.Consequent(np.arange(0, 101, 1), 'assis')
        penche = ctrl.Consequent(np.arange(0, 101, 1), 'penche')
        
        for var in [debout, assis, penche]:
            var['faible'] = fuzz.trimf(var.universe, [0, 0, 40])
            var['moyen'] = fuzz.trimf(var.universe, [30, 50, 70])
            var['fort'] = fuzz.trimf(var.universe, [60, 100, 100])
        
        rules = []
        hr, ta, hkd = inputs.get('hip_ratio'), inputs.get('trunk_angle'), inputs.get('hip_knee_dist')
        
        if hr and ta:
            rules.extend([
                ctrl.Rule(hr['haut'] & ta['vertical'], debout['fort']),
                ctrl.Rule(hr['haut'] & ta['avant'], debout['moyen']),
                ctrl.Rule(hr['bas'] & ta['vertical'], assis['fort'])
            ])
        if hkd:
            rules.extend([
                ctrl.Rule(hkd['grande'], debout['fort']),
                ctrl.Rule(hkd['petite'], assis['moyen'])
            ])
        if ta:
            rules.extend([
                ctrl.Rule(ta['avant'], penche['moyen']),
                ctrl.Rule(ta['arriere'], penche['faible'])
            ])
        if hr and ta:
            rules.append(ctrl.Rule(hr['haut'] & ta['avant'], penche['fort']))
        
        system = ctrl.ControlSystem(rules)
        return {'sim': ctrl.ControlSystemSimulation(system), 'inputs': inputs}

    def compute(self, hip_ratio=None, trunk_angle=None, hip_knee_dist=None):
        try:
            if hip_ratio is not None and trunk_angle is not None and hip_knee_dist is not None:
                system = self.system_full
                mode = "complet"
            elif hip_ratio is not None and trunk_angle is not None:
                system = self.system_partial
                mode = "partiel"
            elif trunk_angle is not None:
                system = self.system_minimal
                mode = "minimal"
            else:
                return {'debout': 33.3, 'assis': 33.3, 'penche': 33.3, 'mode': 'echec'}

            sim = system['sim']
            inputs = system['inputs']

            if 'hip_ratio' in inputs and hip_ratio is not None:
                sim.input['hip_ratio'] = np.clip(hip_ratio, 0.0, 1.0)
            if 'trunk_angle' in inputs and trunk_angle is not None:
                sim.input['trunk_angle'] = np.clip(trunk_angle, -90.0, 90.0)
            if 'hip_knee_dist' in inputs and hip_knee_dist is not None:
                sim.input['hip_knee_dist'] = np.clip(hip_knee_dist, 0.0, 200.0)

            sim.compute()
            return {
                'debout': float(sim.output.get('debout', 0)),
                'assis': float(sim.output.get('assis', 0)),
                'penche': float(sim.output.get('penche', 0)),
                'mode': mode
            }
        except:
            return {'debout': 33.3, 'assis': 33.3, 'penche': 33.3, 'mode': 'erreur'}
    
    def get_label(self, scores):
        mode = scores.pop('mode', 'complet')
        total = sum(scores.values())
        
        if total == 0 or any(np.isnan(v) for v in scores.values()):
            scores = {'debout': 33.3, 'assis': 33.3, 'penche': 33.3}
            total = 100.0
        
        norm = {k: v / total * 100 for k, v in scores.items()}
        main = sorted(norm.items(), key=lambda x: x[1], reverse=True)
        label = f"{main[0][0].capitalize()} ({int(main[0][1])}%)"
        
        if len(main) > 1 and main[1][1] > 15:
            label += f" + {main[1][0].capitalize()} ({int(main[1][1])}%)"

        if mode == "partiel":
            label += " ~"
        elif mode == "minimal":
            label += " ?"

        return label


def analyze_face_emotion(frame, bbox, track_id, emotion_history, fps):
    """
    Analyse l'émotion faciale à partir d'une bounding box.
    Utilise un historique pour lisser les résultats.
    """
    x1, y1, x2, y2 = map(int, bbox)
    # Vérifier que la ROI est valide
    if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0:
        return "neutral", 0.0

    face_roi = frame[y1:y2, x1:x2]
    if face_roi.size == 0:
        return "neutral", 0.0

    try:
        result = DeepFace.analyze(
            face_roi,
            actions=['emotion'],
            enforce_detection=False,
            silent=True  # Désactive les logs
        )
        emotion = result[0]['dominant_emotion']
        confidence = result[0]['emotion'][emotion]
    except:
        emotion = "neutral"
        confidence = 0.0

    if track_id not in emotion_history:
        emotion_history[track_id] = []
    emotion_history[track_id].append((emotion, confidence))
    if len(emotion_history[track_id]) > 5:
        emotion_history[track_id].pop(0)

    emotions = [e for e, _ in emotion_history[track_id]]
    if not emotions:
        return "neutral", 0.0
    dominant_emotion = max(set(emotions), key=emotions.count)
    avg_conf = np.mean([c for e, c in emotion_history[track_id] if e == dominant_emotion])
    return dominant_emotion, avg_conf
class HandFuzzyClassifier:
    def __init__(self):
        import skfuzzy as fuzz
        from skfuzzy import control as ctrl

        self.wrist_dist = ctrl.Antecedent(np.arange(0, 301, 1), 'wrist_dist')
        self.fingertip_dist = ctrl.Antecedent(np.arange(0, 151, 1), 'fingertip_dist')
        self.wrist_height = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'wrist_height')

        self.ouvertes = ctrl.Consequent(np.arange(0, 101, 1), 'ouvertes')
        self.fermees = ctrl.Consequent(np.arange(0, 101, 1), 'fermees')
        self.croisees = ctrl.Consequent(np.arange(0, 101, 1), 'croisees')

        self.wrist_dist['proches'] = fuzz.trimf(self.wrist_dist.universe, [0, 0, 70])
        self.wrist_dist['moyennes'] = fuzz.trimf(self.wrist_dist.universe, [60, 120, 180])
        self.wrist_dist['eloignees'] = fuzz.trimf(self.wrist_dist.universe, [160, 300, 300])

        self.fingertip_dist['petite'] = fuzz.trimf(self.fingertip_dist.universe, [0, 0, 50])
        self.fingertip_dist['moyenne'] = fuzz.trimf(self.fingertip_dist.universe, [40, 80, 100])
        self.fingertip_dist['grande'] = fuzz.trimf(self.fingertip_dist.universe, [90, 150, 150])

        self.wrist_height['hautes'] = fuzz.trimf(self.wrist_height.universe, [0, 0, 0.4])
        self.wrist_height['milieu'] = fuzz.trimf(self.wrist_height.universe, [0.3, 0.6, 0.8])
        self.wrist_height['basses'] = fuzz.trimf(self.wrist_height.universe, [0.7, 1.0, 1.0])

        for var in [self.ouvertes, self.fermees, self.croisees]:
            var['faible'] = fuzz.trimf(var.universe, [0, 0, 40])
            var['moyen'] = fuzz.trimf(var.universe, [30, 50, 70])
            var['fort'] = fuzz.trimf(var.universe, [60, 100, 100])

        rules = [
            ctrl.Rule(self.wrist_dist['proches'] & self.wrist_height['milieu'], self.croisees['fort']),
            ctrl.Rule(self.fingertip_dist['petite'], self.fermees['fort']),
            ctrl.Rule(self.fingertip_dist['grande'] & self.wrist_dist['eloignees'], self.ouvertes['fort']),
        ]

        self.system = ctrl.ControlSystem(rules)
        self.sim = ctrl.ControlSystemSimulation(self.system)

    def compute(self, wrist_dist, fingertip_dist, wrist_height):
        try:
            self.sim.input['wrist_dist'] = np.clip(wrist_dist, 0, 300)
            self.sim.input['fingertip_dist'] = np.clip(fingertip_dist, 0, 150)
            self.sim.input['wrist_height'] = np.clip(wrist_height, 0, 1.0)
            self.sim.compute()
            return {
                'ouvertes': float(self.sim.output.get('ouvertes', 33.3)),
                'fermees': float(self.sim.output.get('fermees', 33.3)),
                'croisees': float(self.sim.output.get('croisees', 33.3))
            }
        except:
            return {'ouvertes': 33.3, 'fermees': 33.3, 'croisees': 33.3}

    def get_label(self, scores):
        total = sum(scores.values())
        if total == 0:
            return "Stable"
        norm = {k: v / total * 100 for k, v in scores.items()}
        main = max(norm.items(), key=lambda x: x[1])
        return f"{main[0].capitalize()} ({int(main[1])}%)"



def get_posture_label(kpts, frame_height, track_id, posture_history, posture_classifier, fps):
    hip_ratio = trunk_angle = hip_knee_dist = None

    left_hip = safe_kpt(kpts, 11)
    right_hip = safe_kpt(kpts, 12)
    if left_hip and right_hip:
        mid_hip_y = (left_hip[1] + right_hip[1]) / 2
        hip_ratio = mid_hip_y / frame_height

    left_shoulder = safe_kpt(kpts, 5)
    right_shoulder = safe_kpt(kpts, 6)
    if left_shoulder and right_shoulder and left_hip and right_hip:
        sh_x = (left_shoulder[0] + right_shoulder[0]) / 2
        sh_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_x = (left_hip[0] + right_hip[0]) / 2
        hip_y = (left_hip[1] + right_hip[1]) / 2
        dx = hip_x - sh_x
        dy = hip_y - sh_y
        trunk_angle = np.degrees(np.arctan2(dx, dy))

    left_knee = safe_kpt(kpts, 13)
    right_knee = safe_kpt(kpts, 14)
    if left_knee and right_knee and left_hip and right_hip:
        knee_y = (left_knee[1] + right_knee[1]) / 2
        hip_y = (left_hip[1] + right_hip[1]) / 2
        hip_knee_dist = abs(hip_y - knee_y)

    scores = posture_classifier.compute(hip_ratio, trunk_angle, hip_knee_dist)

    if track_id not in posture_history:
        posture_history[track_id] = []
    posture_history[track_id].append(scores)
    if len(posture_history[track_id]) > 7:
        posture_history[track_id].pop(0)
    
    avg_scores = {}
    for key in ['debout', 'assis', 'penche']:
        avg_scores[key] = np.mean([s[key] for s in posture_history[track_id] if key in s])
    avg_scores['mode'] = scores.get('mode', 'complet')
    
    return posture_classifier.get_label(avg_scores)






def analyze_arm_posture(kpts_clean, frame_shape):
    """
    Analyse la posture des bras à partir des keypoints YOLO Pose.
    Retourne une étiquette comportementale.
    """
    left_shoulder = safe_kpt(kpts_clean, 5)
    right_shoulder = safe_kpt(kpts_clean, 6)
    left_elbow = safe_kpt(kpts_clean, 7)
    right_elbow = safe_kpt(kpts_clean, 8)
    left_wrist = safe_kpt(kpts_clean, 9)
    right_wrist = safe_kpt(kpts_clean, 10)
    
    if not all([left_shoulder, right_shoulder, left_wrist, right_wrist]):
        return "Inconnu", {"croise": 0, "ouvert": 0, "cache": 0}
    
    h, w = frame_shape[:2]
    
    wrist_dist = np.linalg.norm(np.array(left_wrist) - np.array(right_wrist))
    shoulder_width = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))
    wrist_ratio = wrist_dist / (shoulder_width + 1e-6)  # éviter division par 0
    
    avg_wrist_y = (left_wrist[1] + right_wrist[1]) / 2
    avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
    wrist_above_shoulders = avg_wrist_y < avg_shoulder_y  # mains levées ?
    
    torso_center_x = (left_shoulder[0] + right_shoulder[0] + 
                      (safe_kpt(kpts_clean, 11)[0] if safe_kpt(kpts_clean, 11) else left_shoulder[0]) +
                      (safe_kpt(kpts_clean, 12)[0] if safe_kpt(kpts_clean, 12) else right_shoulder[0])) / 4
    wrist_center_x = (left_wrist[0] + right_wrist[0]) / 2
    wrists_in_front = abs(wrist_center_x - torso_center_x) < 0.6 * shoulder_width
    
    hip_y = (safe_kpt(kpts_clean, 11)[1] if safe_kpt(kpts_clean, 11) else h)
    hands_hidden = avg_wrist_y > hip_y + 0.1 * h
    
    scores = {"croise": 0, "ouvert": 0, "cache": 0}
    
    if wrist_ratio < 0.6 and wrists_in_front and not wrist_above_shoulders:
        scores["croise"] = 90
        label = "Bras croisés"
    elif wrist_ratio > 1.0 and not hands_hidden:
        # Bras écartés → ouverture
        scores["ouvert"] = 85
        label = "Bras ouverts"
    elif hands_hidden:
        scores["cache"] = 80
        label = "Mains cachées"
    elif wrist_above_shoulders:
        scores["ouvert"] = 70
        label = "Mains levées"
    else:
        label = "Neutre"
    
    return label, scores

class ArmPostureFuzzyClassifier:
    def __init__(self):
        import skfuzzy as fuzz
        from skfuzzy import control as ctrl

        wrist_ratio = ctrl.Antecedent(np.arange(0, 2.01, 0.01), 'wrist_ratio')
        wrists_in_front = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'wrists_in_front')  # bool → 0/1
        hands_hidden = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'hands_hidden')
        wrist_above_shoulders = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'wrist_above_shoulders')

        croisees = ctrl.Consequent(np.arange(0, 101, 1), 'croisees')
        ouvertes = ctrl.Consequent(np.arange(0, 101, 1), 'ouvertes')
        cachees = ctrl.Consequent(np.arange(0, 101, 1), 'cachees')
        levees = ctrl.Consequent(np.arange(0, 101, 1), 'levees')

        wrist_ratio['proche'] = fuzz.trimf(wrist_ratio.universe, [0, 0, 0.6])
        wrist_ratio['moyen'] = fuzz.trimf(wrist_ratio.universe, [0.5, 1.0, 1.5])
        wrist_ratio['eloigne'] = fuzz.trimf(wrist_ratio.universe, [1.4, 2.0, 2.0])

        for ant in [wrists_in_front, hands_hidden, wrist_above_shoulders]:
            ant['non'] = fuzz.trimf(ant.universe, [0, 0, 0.5])
            ant['oui'] = fuzz.trimf(ant.universe, [0.5, 1.0, 1.0])

        for conseq in [croisees, ouvertes, cachees, levees]:
            conseq['faible'] = fuzz.trimf(conseq.universe, [0, 0, 40])
            conseq['moyen'] = fuzz.trimf(conseq.universe, [30, 50, 70])
            conseq['fort'] = fuzz.trimf(conseq.universe, [60, 100, 100])

        rules = [
            ctrl.Rule(wrist_ratio['proche'] & wrists_in_front['oui'] & wrist_above_shoulders['non'], croisees['fort']),
            ctrl.Rule(wrist_ratio['eloigne'] & hands_hidden['non'], ouvertes['fort']),
            ctrl.Rule(hands_hidden['oui'], cachees['fort']),
            ctrl.Rule(wrist_above_shoulders['oui'], levees['fort']),
        ]

        self.system = ctrl.ControlSystem(rules)
        self.sim = ctrl.ControlSystemSimulation(self.system)

    def compute(self, wrist_ratio, wrists_in_front, hands_hidden, wrist_above_shoulders):
        try:
            self.sim.input['wrist_ratio'] = np.clip(wrist_ratio, 0, 2.0)
            self.sim.input['wrists_in_front'] = float(wrists_in_front)
            self.sim.input['hands_hidden'] = float(hands_hidden)
            self.sim.input['wrist_above_shoulders'] = float(wrist_above_shoulders)
            self.sim.compute()
            return {
                'croisees': float(self.sim.output.get('croisees', 25)),
                'ouvertes': float(self.sim.output.get('ouvertes', 25)),
                'cachees': float(self.sim.output.get('cachees', 25)),
                'levees': float(self.sim.output.get('levees', 25))
            }
        except:
            return {'croisees': 25, 'ouvertes': 25, 'cachees': 25, 'levees': 25}

    def get_label(self, scores):
        total = sum(scores.values())
        if total == 0 or any(np.isnan(v) for v in scores.values()):
            return "Neutre"
        norm = {k: v / total * 100 for k, v in scores.items()}
        main = max(norm.items(), key=lambda x: x[1])
        return f"{main[0].capitalize()} ({int(main[1])}%)"
    

class AgitationFuzzyClassifier:
    def __init__(self):
        import skfuzzy as fuzz
        from skfuzzy import control as ctrl

        wrist_speed = ctrl.Antecedent(np.arange(0, 50, 0.5), 'wrist_speed')  # pixels/frame
        agitation = ctrl.Consequent(np.arange(0, 101, 1), 'agitation')

        wrist_speed['lente'] = fuzz.trimf(wrist_speed.universe, [0, 0, 5])
        wrist_speed['moderee'] = fuzz.trimf(wrist_speed.universe, [4, 10, 20])
        wrist_speed['rapide'] = fuzz.trimf(wrist_speed.universe, [18, 50, 50])

        agitation['calme'] = fuzz.trimf(agitation.universe, [0, 0, 40])
        agitation['moderee'] = fuzz.trimf(agitation.universe, [30, 50, 70])
        agitation['agitee'] = fuzz.trimf(agitation.universe, [60, 100, 100])

        rules = [
            ctrl.Rule(wrist_speed['lente'], agitation['calme']),
            ctrl.Rule(wrist_speed['moderee'], agitation['moderee']),
            ctrl.Rule(wrist_speed['rapide'], agitation['agitee'])
        ]

        self.system = ctrl.ControlSystem(rules)
        self.sim = ctrl.ControlSystemSimulation(self.system)

    def compute(self, wrist_speed):
        try:
            self.sim.input['wrist_speed'] = np.clip(wrist_speed, 0, 50)
            self.sim.compute()
            score = float(self.sim.output['agitation'])
            if score < 40:
                return "Calme", score
            elif score < 70:
                return "Modérée", score
            else:
                return "Agitée", score
        except:
            return "Inconnue", 0

def compute_wrist_speed(kpts_clean, prev_kpts, fps):
    """
    Calcule la vitesse moyenne des poignets en pixels/seconde.
    """
    if prev_kpts is None:
        return 0.0

    left_wrist = safe_kpt(kpts_clean, 9)
    right_wrist = safe_kpt(kpts_clean, 10)
    prev_left = safe_kpt(prev_kpts, 9)
    prev_right = safe_kpt(prev_kpts, 10)

    speed = 0.0
    count = 0

    if left_wrist and prev_left:
        dist = np.linalg.norm(np.array(left_wrist) - np.array(prev_left))
        speed += dist * fps
        count += 1
    if right_wrist and prev_right:
        dist = np.linalg.norm(np.array(right_wrist) - np.array(prev_right))
        speed += dist * fps
        count += 1

    return speed / count if count > 0 else 0.0

def main():
    st.markdown('<h1 class="main-header">🔍 Analyse Comportementale en Temps Réel</h1>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        video_source = st.text_input("📹 Chemin vidéo", "vdfi.mp4")
        playback_speed = st.slider("⏩ Vitesse de lecture", 0.2, 3.0, 1.0, 0.2)
        skip_every = st.slider("⏩ Frames à sauter", 0, 5, 1)  # 0 = tout traiter, 1 = 1/2, 2 = 1/3, etc.
        st.markdown("---")
        st.subheader("🎨 Options de visualisation")
        show_confidence = st.checkbox("📊 Afficher confiance", value=True)
        show_pose_skeleton = st.checkbox("🦴 Squelette du corps", value=True)
        skeleton_thickness = st.slider("Épaisseur du squelette", 5, 20, 7)#1,5,2
        
        st.markdown("---")
        st.subheader("🎯 Légende États")
        st.markdown("""
        - ✅ **Confiant** : Bras ouverts
        - ❌ **Méfiant / Stressé** : Bras croisés
        - ⚠️ **Fermé / Nerveux** : Mains cachées
        - 🙂 **Neutre** : Aucun signe particulier
        """)
        
        st.markdown("---")
        st.subheader("🎨 Code couleur keypoints")
        st.markdown("""
        **Corps:**
        - 🟣 Nez
        - 🔵 Yeux
        - 🟢 Épaules
        - 🔴 Hanches
        - 🟠 Poignets
        
        **Mains:**
        - 🟣 Poignet
        - 🟡 Bouts des doigts
        """)
    
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'posture_history' not in st.session_state:
        st.session_state.posture_history = {}
    if 'prev_keypoints' not in st.session_state:
        st.session_state.prev_keypoints = {}
    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = {}
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        start_btn = st.button("▶️ Démarrer", use_container_width=True)
    with col2:
        stop_btn = st.button("⏹️ Arrêter", use_container_width=True)
    
    if start_btn:
        st.session_state.running = True
    if stop_btn:
        st.session_state.running = False
    
    metrics_placeholder = st.empty()
    table_placeholder = st.empty()
    video_placeholder = st.empty()
    
    if st.session_state.running:
        with st.spinner("🔄 Chargement des modèles..."):
            model = YOLO("yolov8n-pose.pt")
            posture_classifier = PostureFuzzyClassifier()
            
           
        
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            st.error(f"❌ Impossible d'ouvrir '{video_source}'")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = int(1000 / fps / playback_speed)
        
        st.success(f"✅ Vidéo chargée : {fps:.1f} fps")
        
        frame_count = 0
        while st.session_state.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.session_state.running = False
                break
            
            frame_count += 1

            
         
            track_data = []
            
           
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.25, imgsz=320,vid_stride=2)

            annotated_frame = frame.copy()  

            if hasattr(model.predictor.args, 'tracker_type'):
                print("Tracker type in default config:", model.predictor.args.tracker_type)
            else:
                print("Tracker type not found in default config, using default tracker.")
            
            if results[0].keypoints.conf is not None and len(results[0].keypoints.conf) > 0:
                
                
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    kpts_list = results[0].keypoints.xy.cpu().numpy()
                    for i, track_id in enumerate(track_ids):
                        if i >= len(kpts_list):
                            continue

                        box_conf = float(results[0].boxes.conf[i].cpu().numpy()) if i < len(results[0].boxes.conf) else 0.0
                        if box_conf < 0.1:
                            continue

                        kpts = kpts_list[i]
                        kpts_clean = [
                            (int(pt[0]), int(pt[1])) if pt[0] > 0 and pt[1] > 0 else None 
                            for pt in kpts
                        ]

                        if not is_valid_human_pose(kpts_clean, frame.shape):
                            continue

                        if results[0].boxes.xyxy.shape[0] <= i:
                            continue
                        x1, y1, x2, y2 = map(int, results[0].boxes.xyxy[i].cpu().numpy())

                      
                        face_emotion, face_conf = analyze_face_emotion(
                            frame, [x1, y1, x2, y2], track_id,
                            st.session_state.emotion_history, fps
                        )

                        left_shoulder = safe_kpt(kpts_clean, 5)
                        right_shoulder = safe_kpt(kpts_clean, 6)
                        left_wrist = safe_kpt(kpts_clean, 9)
                        right_wrist = safe_kpt(kpts_clean, 10)
                        left_hip = safe_kpt(kpts_clean, 11)
                        right_hip = safe_kpt(kpts_clean, 12)

                        if not all([left_shoulder, right_shoulder, left_wrist, right_wrist]):
                            arm_label = "Inconnu"
                            agitation_label = "Inconnue"
                        else:
                            h, w = frame.shape[:2]
                            shoulder_width = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))
                            wrist_dist = np.linalg.norm(np.array(left_wrist) - np.array(right_wrist))
                            wrist_ratio = wrist_dist / (shoulder_width + 1e-6)

                            avg_wrist_y = (left_wrist[1] + right_wrist[1]) / 2
                            avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
                            wrist_above_shoulders = avg_wrist_y < avg_shoulder_y

                            torso_center_x = (left_shoulder[0] + right_shoulder[0] +
                                            (left_hip[0] if left_hip else left_shoulder[0]) +
                                            (right_hip[0] if right_hip else right_shoulder[0])) / 4
                            wrist_center_x = (left_wrist[0] + right_wrist[0]) / 2
                            wrists_in_front = abs(wrist_center_x - torso_center_x) < 0.6 * shoulder_width

                            hip_y = left_hip[1] if left_hip else h
                            hands_hidden = avg_wrist_y > hip_y + 0.1 * h

                            arm_classifier = ArmPostureFuzzyClassifier()
                            arm_scores = arm_classifier.compute(
                                wrist_ratio=wrist_ratio,
                                wrists_in_front=wrists_in_front,
                                hands_hidden=hands_hidden,
                                wrist_above_shoulders=wrist_above_shoulders
                            )
                            arm_label = arm_classifier.get_label(arm_scores)

                            prev_kpts = st.session_state.prev_keypoints.get(track_id, None)
                            wrist_speed_px_per_sec = compute_wrist_speed(kpts_clean, prev_kpts, fps)
                            agitation_classifier = AgitationFuzzyClassifier()
                            agitation_label, agitation_score = agitation_classifier.compute(wrist_speed_px_per_sec / fps)
                            st.session_state.prev_keypoints[track_id] = kpts_clean

                        croisees_score = arm_scores.get("croisees", 0)
                        ouvertes_score = arm_scores.get("ouvertes", 0)
                        cachees_score = arm_scores.get("cachees", 0)

                        if (croisees_score > 70) and (face_emotion in ["angry", "fear", "sad"]):
                            psychological_state = "Stressé / Anxieux"
                            state_emoji = "😰"
                        elif (ouvertes_score > 60) and (face_emotion == "happy"):
                            psychological_state = "Confiant / Positif"
                            state_emoji = "😊"
                        elif (cachees_score > 60) and (face_emotion in ["neutral", "sad"]):
                            psychological_state = "Fermé / Nerveux"
                            state_emoji = "😐"
                        elif face_emotion == "surprise":
                            psychological_state = "Étonné"
                            state_emoji = "😮"
                        else:
                            if croisees_score > 70:
                                psychological_state = "Méfiant / Stressé"
                                state_emoji = "❌"
                            elif ouvertes_score > 60:
                                psychological_state = "Confiant"
                                state_emoji = "✅"
                            elif cachees_score > 60:
                                psychological_state = "Fermé / Nerveux"
                                state_emoji = "⚠️"
                            else:
                                psychological_state = "Neutre"
                                state_emoji = "🙂"

                        posture_label = get_posture_label(
                            kpts_clean, frame.shape[0], int(track_id),
                            st.session_state.posture_history, posture_classifier, fps
                        )

                        duration_sec = len(st.session_state.posture_history.get(track_id, [])) / fps

                        color_map = {
                            "😰": (0, 0, 255),      
                            "😊": (0, 255, 0),     
                            "😐": (255, 165, 0),   
                            "😮": (255, 255, 0),   
                            "❌": (0, 0, 255),
                            "✅": (0, 255, 0),
                            "⚠️": (255, 165, 0),
                            "🙂": (255, 255, 255)   
                        }
                        bbox_color = color_map.get(state_emoji, (255, 255, 255))

                       
                        track_data.append({
                            'ID': f"👤 {track_id}",
                            'Posture': posture_label,
                            'Bras/Mains': arm_label,
                            'Émotion': f"{face_emotion.capitalize()} ({int(face_conf)}%)",
                            'Agitation': agitation_label,
                            'État': f"{state_emoji} {psychological_state}",
                            'Durée (s)': f"{duration_sec:.1f}",
                            'Confiance': f"{box_conf:.2f}"
})

                        if show_pose_skeleton:
                            draw_pose_keypoints(annotated_frame, kpts, track_id, bbox_color, skeleton_thickness)

                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), bbox_color, 2)
                        
                        label = f"ID{track_id}"
                        if show_confidence:
                            label += f" {box_conf:.2f}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(annotated_frame, 
                                    (x1, y1 - label_size[1] - 10),
                                    (x1 + label_size[0] + 10, y1),
                                    bbox_color, -1)
                        cv2.putText(annotated_frame, label,
                                    (x1 + 5, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 4,
                                    (0, 0, 0), 4, cv2.LINE_AA)
               
            if track_data:
                with metrics_placeholder.container():
                    cols = st.columns(4)
                    cols[0].metric("👥 Personnes détectées", len(track_data))
                    
                    stress_count = sum(1 for d in track_data if "Stress" in d['État'] or "Anxiété" in d['État'])
                    cols[1].metric("⚠️ Alertes", stress_count)
                    
                    avg_conf = np.mean([float(d['Confiance']) for d in track_data])
                    cols[2].metric("📊 Confiance moyenne", f"{avg_conf:.2f}")
                    
                    cols[3].metric("🎬 Frame", frame_count)
                
                df = pd.DataFrame(track_data)
                table_placeholder.dataframe(df, use_container_width=True, hide_index=True)
            
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
            
        
        cap.release()
        st.success(" Analyse terminée !")

if __name__ == "__main__":
    main()