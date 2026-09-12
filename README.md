# 🔍 Analyse Comportementale

> **Vision par ordinateur** pour l'analyse en temps réel du comportement humain.

---

## 🪧 À propos

Ce projet regroupe **deux applications distinctes** de vision par ordinateur qui analysent le comportement de personnes à partir d'une webcam ou d'une vidéo enregistrée. Elles ne partagent pas le même moteur de détection :

| | `app.py` (webcam live) | `detect.py` (vidéo enregistrée) |
| :--- | :--- | :--- |
| **Détection** | MediaPipe Holistic (pose + visage + mains) | YOLOv8-pose (`ultralytics`) avec suivi ByteTrack |
| **Ce qui est mesuré** | Regard (attentif/distrait), posture (bonne/mauvaise), durée | Posture (floue), position des bras, agitation (vitesse des poignets), émotion faciale (DeepFace), état psychologique estimé |
| **Export** | CSV des statistiques | — (pas d'export pour l'instant) |

**Pourquoi ce projet ?**
- Détecter la posture pour prévenir les problèmes de dos
- Analyser le regard et l'agitation pour mesurer l'attention/le stress
- Explorer MediaPipe, YOLO-pose, DeepFace et la logique floue (`scikit-fuzzy`) pour la classification comportementale
- Créer une interface simple et interactive avec Streamlit

⚠️ **Note** : les deux scripts sont indépendants — ce n'est pas un "mode 1 / mode 2" du même moteur, mais deux prototypes différents dans le même dépôt.

---
## 📸 Captures d'écran

![image alt](https://github.com/abidirymeh/analyse-comportementale/blob/57455a820ab7bb3c7de7d5fc6a6a2d33e264d8b9/capture1.jfif)
![image alt](https://github.com/abidirymeh/analyse-comportementale/blob/57455a820ab7bb3c7de7d5fc6a6a2d33e264d8b9/capture2.jfif)
![image alt](https://github.com/abidirymeh/analyse-comportementale/blob/57455a820ab7bb3c7de7d5fc6a6a2d33e264d8b9/capture3.jfif)
![image alt](https://github.com/abidirymeh/analyse-comportementale/blob/57455a820ab7bb3c7de7d5fc6a6a2d33e264d8b9/capture4.jfif)
![image alt](https://github.com/abidirymeh/analyse-comportementale/blob/57455a820ab7bb3c7de7d5fc6a6a2d33e264d8b9/capture5.jfif)

---

## 📦 Prérequis

| Élément | Version minimale | Lien |
| :--- | :--- | :--- |
| **Python** | 3.10+ | [python.org](https://python.org) |
| **pip** | 22.0+ | [pip documentation](https://pip.pypa.io/) |
| **Git** | 2.0+ | [git-scm.com](https://git-scm.com/) |

---

## 🚀 Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/abidirymeh/analyse-comportementale.git
cd analyse-comportementale
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
```

### 3. Activer l'environnement virtuel

**Windows :**
```bash
venv\Scripts\activate
```

**Mac / Linux :**
```bash
source venv/bin/activate
```

### 4. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 5. Vérifier l'installation

```bash
pip list
```

###
## 🛠️ Utilisation

### Mode 1 : Webcam en direct — `app.py` (MediaPipe)

```bash
streamlit run app.py
```

Ouvre ensuite ton navigateur à : http://localhost:8501

Fonctionnalités :
- ✅ Capture et analyse en temps réel via WebRTC
- ✅ Détection du regard (attentif / distrait) et de la posture
- ✅ Tableau de bord interactif dans la barre latérale
- ✅ Export des statistiques en CSV

### Mode 2 : Vidéo enregistrée — `detect.py` (YOLO-pose + DeepFace + logique floue)

```bash
streamlit run detect.py
```

Dans la barre latérale, configure :
- **Chemin vidéo** 
- **Vitesse de lecture** et **frames à sauter**
- Affichage de la confiance et du squelette (avec épaisseur réglable)

Puis clique sur **▶️ Démarrer** (et **⏹️ Arrêter** pour stopper).

Fonctionnalités :
- ✅ Suivi multi-personnes (ByteTrack)
- ✅ Classification floue de la posture (debout / assis / penché)
- ✅ Détection de la position des bras (croisés / ouverts / cachés)
- ✅ Estimation de l'agitation (vitesse des poignets)
- ✅ Reconnaissance d'émotion faciale (DeepFace)
- ✅ Estimation d'un "état psychologique" combinant ces signaux (ex. 😊 Confiant, 😰 Stressé/Anxieux)

### Mode 3 : Exporter les données (webcam uniquement)

Une fois l'analyse `app.py` en cours, clique sur **"Exporter CSV"** dans la barre latérale.

Format du fichier CSV :
```csv
ID,Attention_%,Mauvaise_posture_%,Durée_s
1,95.2,5.8,12
2,67.3,32.7,8
```

---

## 🤝 Contribution

Les contributions sont les bienvenues !

1. **Forker le projet** — bouton "Fork" en haut à droite du dépôt GitHub.
2. **Cloner ton fork**
   ```bash
   git clone https://github.com/abidirymeh/analyse-comportementale.git
   cd analyse-comportementale
   ```
3. **Créer une branche pour ta feature**
   ```bash
   git checkout -b feature/nom-de-ta-feature
   ```
4. **Faire tes modifications** — ajoute ton code, vérifie que tout fonctionne, mets à jour la doc si besoin.
5. **Commiter tes changements**
   ```bash
   git add .
   git commit -m "Ajout de la fonctionnalité X"
   ```
6. **Pousser ta branche**
   ```bash
   git push origin feature/nom-de-ta-feature
   ```
7. **Ouvrir une Pull Request** vers la branche `main` du dépôt original.

**Conventions de commit :**
- `feat:` pour une nouvelle fonctionnalité
- `fix:` pour une correction de bug
- `docs:` pour la documentation
- `style:` pour le formatage
- `refactor:` pour une refactorisation
- `test:` pour les tests
- `chore:` pour les tâches de maintenance

---

## 🏗️ Construit avec

### Langages & Frameworks

| Outil | Rôle | Lien |
| :--- | :--- | :--- |
| Python 3.10+ | Langage principal | [python.org](https://python.org) |
| Streamlit | Interface utilisateur | [streamlit.io](https://streamlit.io) |
| Streamlit-WebRTC | Capture webcam (`app.py`) | [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) |
| MediaPipe | Détection pose/visage/mains (`app.py`) | [mediapipe.dev](https://mediapipe.dev) |
| Ultralytics YOLOv8-pose | Détection + suivi de pose (`detect.py`) | [ultralytics.com](https://ultralytics.com) |
| DeepFace | Reconnaissance d'émotion faciale (`detect.py`) | [github.com/serengil/deepface](https://github.com/serengil/deepface) |
| scikit-fuzzy | Classification floue posture/bras/agitation (`detect.py`) | [pypi.org/project/scikit-fuzzy](https://pypi.org/project/scikit-fuzzy) |
| OpenCV | Traitement d'images | [opencv.org](https://opencv.org) |
| NumPy | Calculs numériques | [numpy.org](https://numpy.org) |
| Pandas | Gestion des données | [pandas.pydata.org](https://pandas.pydata.org) |

### Outils

| Outil | Rôle | Lien |
| :--- | :--- | :--- |
| Git | Gestion de version | [git-scm.com](https://git-scm.com) |
| GitHub | Hébergement du code | [github.com](https://github.com) |
| VSCode | Éditeur de code | [code.visualstudio.com](https://code.visualstudio.com) |

### CI / CD

Actuellement, ce projet n'utilise pas d'intégration continue. Amélioration future possible.

### Déploiement

Conçu pour fonctionner en local. Un déploiement en ligne est envisageable avec Streamlit Cloud ou Hugging Face Spaces — à condition de vérifier la compatibilité de YOLO/DeepFace avec l'environnement cible (poids de modèles, mémoire, GPU).

| Plateforme | Lien |
| :--- | :--- |
| Streamlit Cloud | [streamlit.io/cloud](https://streamlit.io/cloud) |
| Hugging Face Spaces | [huggingface.co/spaces](https://huggingface.co/spaces) |

---

## 📚 Documentation

### Documentation embarquée
- `README.md` — présentation générale du projet
- `requirements.txt` — liste des dépendances

### Documentation externe

| Sujet | Lien |
| :--- | :--- |
| MediaPipe Pose | [MediaPipe Pose Guide](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) |
| Ultralytics YOLO Pose | [docs.ultralytics.com](https://docs.ultralytics.com/tasks/pose/) |
| DeepFace | [github.com/serengil/deepface](https://github.com/serengil/deepface) |
| scikit-fuzzy | [pythonhosted.org/scikit-fuzzy](https://pythonhosted.org/scikit-fuzzy/) |
| Streamlit Documentation | [docs.streamlit.io](https://docs.streamlit.io) |
| OpenCV Tutorials | [docs.opencv.org](https://docs.opencv.org) |

### Structure du code

```
CV2/
│
├── core/                          # Modules de app.py (à créer, voir Installation §6)
│   ├── detectors.py               # HolisticDetector (MediaPipe)
│   ├── tracker.py                 # PersonTracker
│   ├── analyzers.py               # AttentionAnalyzer, PostureAnalyzer
│   └── visualizer.py              # Visualizer (overlays OpenCV)
│
├                     
├── RESULT.mp4                     # Exemple de vidéo pour detect.py
│
├── app.py                         # App webcam live (MediaPipe)
├── src/
│     └── detect.py                # App vidéo (YOLO-pose + DeepFace + logique floue)
│                         
├── requirements.txt               # Dépendances
├── README.md                      # Documentation
└── .gitignore                     # Fichiers ignorés
```

---

## 🏷️ Gestion des versions

Le nommage des versions suit la [Gestion sémantique de version](https://semver.org/lang/fr/) : `MAJOR.MINOR.PATCH`

- **MAJOR** : changements incompatibles avec les versions antérieures
- **MINOR** : ajout de fonctionnalités (rétrocompatible)
- **PATCH** : corrections de bugs (rétrocompatible)

Les versions et journaux de changements sont disponibles depuis la page des Releases.

---

## 📝 Licence

Voir le fichier `LICENSE` du dépôt. Ce projet est sous licence MIT — vous pouvez l'utiliser, le modifier et le distribuer librement.

---

## 📧 Contact

- **Auteur** : [Rimeh Abidi]
- **Email** : [rimeh.abidi@enis.tn]
- **GitHub** : https://github.com/abidirymeh
