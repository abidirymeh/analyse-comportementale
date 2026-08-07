# app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import pandas as pd
import time
from collections import defaultdict

# ←←← CES IMPORTS MARCHENT UNIQUEMENT SI LE DOSSIER core EXISTE
from core.detectors import HolisticDetector
from core.tracker import PersonTracker
from core.analyzers import AttentionAnalyzer, PostureAnalyzer
from core.visualizer import Visualizer

st.set_page_config(page_title="Analyse Corporelle Live", layout="wide")
st.title("Analyse Corporelle & Comportement en Temps Réel")

# État global
if "logs" not in st.session_state:
    st.session_state.logs = defaultdict(lambda: {
        "attention_frames": 0, "total_frames": 0,
        "bad_posture_frames": 0, "birth_time": time.time()
    })

table_placeholder = st.sidebar.table(pd.DataFrame(columns=["ID", "Attention", "Posture", "Regard", "Durée (s)"]))

class BodyVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = HolisticDetector()
        self.tracker = PersonTracker()
        self.attention_analyzer = AttentionAnalyzer()
        self.posture_analyzer = PostureAnalyzer()
        self.visualizer = Visualizer()
        self.frame_count = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        results = self.detector.process(img)
        tracked_people = self.tracker.update(results, img)

        rows = []

        for person in tracked_people:
            pid = person.id

            # Attention (regard)
            looking = self.attention_analyzer.is_looking_at_screen(person, w, h)
            log = st.session_state.logs[pid]
            log["total_frames"] += 1
            if looking:
                log["attention_frames"] += 1
            attention_pct = (log["attention_frames"] / max(log["total_frames"], 1)) * 100

            # Posture
            posture = self.posture_analyzer.evaluate(person)
            if "Mauvaise" in posture:
                log["bad_posture_frames"] += 1

            rows.append({
                "ID": pid,
                "Attention": f"{attention_pct:.0f}%",
                "Posture": posture,
                "Regard": "Écran" if looking else "Hors champ",
                "Durée (s)": int(time.time() - log["birth_time"])
            })

            # Dessin
            img = self.visualizer.draw_all(img, person, looking, posture)

        # Mise à jour tableau
        self.frame_count += 1
        if self.frame_count % 10 == 0 and rows:
            table_placeholder.table(pd.DataFrame(rows))

        return img

# Lancement webcam
webrtc_streamer(
    key="live",
    video_transformer_factory=BodyVideoTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True
)

# Bouton export
if st.sidebar.button("Exporter CSV"):
    data = []
    for pid, log in st.session_state.logs.items():
        total = max(log["total_frames"], 1)
        data.append({
            "ID": pid,
            "Attention_%": round((log["attention_frames"]/total)*100, 1),
            "Mauvaise_posture_%": round((log["bad_posture_frames"]/total)*100, 1),
            "Durée_s": int(time.time() - log["birth_time"])
        })
    df = pd.DataFrame(data)
    csv = df.to_csv(index=False).encode()
    st.sidebar.download_button("Télécharger CSV", csv, "analyse_comportement.csv", "text/csv")