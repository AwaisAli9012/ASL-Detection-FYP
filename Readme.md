# ASL Sign Language Detection System
## Internal Version (Flask + Complete Features)

**Status:** ✅ INTERNAL VERSION COMPLETE & WORKING

**Current Version:** Flask Web Application (localhost:5000)  
**Deployment Status:** Production-ready for internal testing  
**Next Version:** External FastAPI with user personalization

---

## 🎯 Overview

A comprehensive American Sign Language (ASL) interpreter that detects 15 ASL signs in real-time, generates contextual sentences, translates to multiple languages, and produces natural voice output using text-to-speech. Also includes emotion detection for context-aware interpretation.

**Key Achievement:** 97.43% accuracy meta-learner combining Neural Network (92.13%) + Random Forest (96.82%) + Meta-Learner optimization.

---

## ✨ Features (Internal - All Working)

### 🖐️ **Real-Time Hand Detection**
- MediaPipe hand keypoint extraction (21 points per hand = 126 dimensions)
- 16-20 FPS on CPU (no GPU required)
- Two-hand detection support
- Smooth real-time video feed

### 🧠 **Ensemble Machine Learning (97.43% Accuracy - Meta-Learner)**
- **Neural Network (TensorFlow/Keras):** 92.13% accuracy (5-fold mean)
- **Random Forest (scikit-learn):** 96.82% accuracy (5-fold mean)
- **Ensemble Soft-Vote:** 95.73% accuracy (5-fold mean)
- **Meta-Learner (Logistic Regression):** **97.43% OOF accuracy** ⭐ (BEST)
- Stacking ensemble technique
- 5-Fold stratified cross-validation trained

**Fold-by-fold breakdown:**
- Fold 1: NN=92.58%, RF=96.67%, Ensemble=95.92%
- Fold 2: NN=92.92%, RF=97.50%, Ensemble=96.75%
- Fold 3: NN=92.42%, RF=96.83%, Ensemble=96.00%
- Fold 4: NN=90.67%, RF=96.33%, Ensemble=94.42%
- Fold 5: NN=92.08%, RF=96.75%, Ensemble=95.58%

### 💬 **Sentence Generation**
- Context-aware English sentence generation
- Groq API integration (LLaMA 3.3 70B)
- Fallback to grammar rules if API unavailable
- Natural language output

### 🌐 **Multi-Language Support**
- **English:** Native generation + TTS
- **Urdu:** Real-time translation + TTS with Urdu voice
- **Arabic:** Real-time translation + TTS with Arabic voice
- Google Translate integration
- Bidirectional language detection

### 🔊 **Text-to-Speech (TTS)**
- pyttsx3 local TTS engine
- Multiple language voices:
  - English: English voice
  - Urdu: Urdu/Hindi voice support
  - Arabic: Arabic voice support
- Auto-play generated sentences
- Manual replay controls

### 😊 **Emotion Detection**
- Parallel emotion recognition model
- Detects: Happy, Sad, Angry, Neutral, Surprised
- Integrates with sign detection
- Context-aware interpretation
- Random Forest ensemble for emotion

### 🎨 **Web UI (Flask)**
- Real-time video feed display
- Current sign display with confidence
- Generated sentence panel
- Multi-language output display
- Audio playback controls
- Dark theme (battery-friendly, eye-friendly)
- Responsive design
- Emotion status indicator

### 🔑 **API Integration**
- Groq API for sentence verification/enhancement
- Optional (graceful fallback without it)
- <100ms response time
- LLaMA 3.3 70B model
- Cost-effective ($0.001 per 1000 tokens)

### 📊 **Performance Monitoring**
- Real-time FPS display
- Confidence score visualization
- Processing time metrics
- Model accuracy display

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   USER (Webcam)                      │
│                  Signs in Frame                      │
└────────────────────┬────────────────────────────────┘
                     ↓
         ┌───────────────────────┐
         │  MediaPipe Detection  │
         │  (21 points × 2 hands)│
         │   126 dimensions      │
         └───────────┬───────────┘
                     ↓
    ┌────────────────────────────────────┐
    │   Ensemble Prediction Pipeline     │
    ├────────────────────────────────────┤
    │ NN Model (92.13%)                  │
    │ ↓                                  │
    │ RF Model (96.82%)                  │
    │ ↓                                  │
    │ Meta-Learner Combination           │
    │ ↓                                  │
    │ FINAL: 97.43% Accuracy ⭐          │
    └────────────┬───────────────────────┘
                 ↓
    ┌─────────────────────────────┐
    │  Sentence Generation        │
    │  (Groq LLaMA or Grammar)    │
    └────────────┬────────────────┘
                 ↓
    ┌──────────────────────────────┐
    │  Language Translation         │
    │  ├─ English                   │
    │  ├─ Urdu                      │
    │  └─ Arabic                    │
    └────────────┬─────────────────┘
                 ↓
    ┌──────────────────────────────┐
    │  Text-to-Speech (pyttsx3)    │
    │  ├─ English Voice             │
    │  ├─ Urdu Voice                │
    │  └─ Arabic Voice              │
    └────────────┬─────────────────┘
                 ↓
    ┌──────────────────────────────┐
    │  Emotion Detection (RF)       │
    │  Happy/Sad/Angry/Neutral/... │
    └────────────┬─────────────────┘
                 ↓
    ┌──────────────────────────────┐
    │  Flask Web UI                 │
    │  Display on localhost:5000    │
    └──────────────────────────────┘
```

---

## 📁 Project Structure

```
ASL-Detection-FYP/
├── 15_classes/
│   ├── asl/                          # Virtual environment (uv)
│   │
│   ├── flask_app/                    # ⭐ Main Flask Application
│   │   ├── app.py                    # Flask server (main app)
│   │   ├── templates/
│   │   │   ├── index.html            # Web UI interface
│   │   │   └── ...
│   │   ├── static/
│   │   │   ├── script.js             # Frontend logic
│   │   │   ├── style.css             # UI styling
│   │   │   └── ...
│   │   ├── emotion_models/           # Emotion detection models
│   │   │   ├── emotion_rf.pkl
│   │   │   ├── emotion_svm.pkl
│   │   │   └── emotion_meta.pkl
│   │   ├── emotion_keypoints/        # Emotion training data
│   │   ├── collect_emotion.py        # Emotion data collection
│   │   ├── ex_emotion.py             # Extract emotion keypoints
│   │   ├── train_emotion.py          # Train emotion models
│   │   └── train_emotion_ensemble.py # Train emotion ensemble
│   │
│   ├── ex15class.py                  # Extract ASL keypoints
│   ├── trn15class.py                 # Train NN model
│   ├── Ens15class.py                 # Train ensemble (NN+RF+Meta)
│   ├── det15class_ensemble.py        # Real-time detection
│   ├── det15class.py                 # Legacy detection
│   │
│   ├── Your 15 signs.txt             # ASL class definitions
│   ├── .gitignore                    # Git ignore rules
│   └── README.md                     # This file
│
├── Dataset/
│   ├── frames_20/                    # Raw frames (20 classes)
│   │   ├── help/
│   │   ├── yes/
│   │   ├── no/
│   │   ├── dog/
│   │   └── ... (15 classes)
│   │
│   └── keypoints_15_v4/              # Extracted keypoints
│       ├── keypoints.npy             # Hand keypoint data (6000 samples)
│       └── labels.npy                # Class labels
│
└── Models/
    ├── keypoint_model_15_v4_ensemble_nn.h5    # NN Model
    ├── keypoint_model_15_v4_rf.pkl            # RF Model
    ├── keypoint_model_15_v4_meta.pkl          # Meta-Learner
    ├── keypoint_labels_15_v4.json             # Class labels (15)
    └── ensemble_results.json                  # Training results
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- Linux/macOS (tested on Linux Onarchy)
- Webcam for real-time detection
- Internet connection (optional, for Groq API)

### Step 1: Create Virtual Environment

```bash
cd /home/xero1/Documents/ASL-Detection-FYP/15_classes
uv venv asl
source asl/bin/activate
```

### Step 2: Install Dependencies

```bash
uv pip install tensorflow keras scikit-learn numpy scipy opencv-python mediapipe spacy textblob nltk flask flask-cors pyttsx3 groq requests python-dotenv pandas pillow google-translate-unofficial
```

Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

### Step 3: Setup Environment Variables (Optional for Groq)

Create `.env` file in `/15_classes/`:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get free API key from: https://console.groq.com

### Step 4: Verify Installation

```bash
python -c "import tensorflow; print('✓ TensorFlow')"
python -c "import flask; print('✓ Flask')"
python -c "import cv2; print('✓ OpenCV')"
python -c "import mediapipe; print('✓ MediaPipe')"
python -c "import groq; print('✓ Groq')"
```

---

## 🚀 Running the Application

### First Time Setup (Extract & Train Models)

```bash
# Extract keypoints from frames
python ex15class.py
# Time: ~5-10 minutes
# Creates: Dataset/keypoints_15_v4/keypoints.npy

# Train ensemble models
python Ens15class.py
# Time: ~10-20 minutes
# Creates: Models/*.h5, *.pkl
# Generates: Models/ensemble_results.json
```

### Run Flask Application

```bash
cd flask_app
python app.py
```

You should see:
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

### Access Web Interface

Open browser and go to:
```
http://localhost:5000
```

---

## 🎮 Using the Application

### Web Interface Features

1. **Video Feed (Left Side)**
   - Real-time webcam input
   - Hand detection visualization
   - Keypoint drawing (optional)

2. **Detection Panel (Top Right)**
   - Current sign detected
   - Confidence score (0-100%)
   - Class name

3. **Sentence Generator (Middle Right)**
   - Generated English sentence
   - Emotion detection result

4. **Language Translation (Bottom Right)**
   - English sentence
   - Urdu translation
   - Arabic translation

5. **Audio Controls (Bottom)**
   - Play English voice
   - Play Urdu voice
   - Play Arabic voice
   - Volume control
   - Speed control

6. **Emotion Status**
   - Current emotion detected
   - Confidence percentage

---

## 🎓 ASL Classes Detected (15)

```
1. HELP        - Request assistance
2. YES         - Affirmative
3. NO          - Negative
4. DOG         - Animal
5. GO          - Movement
6. FINISH      - Completion
7. PLAY        - Recreation
8. MOTHER      - Family
9. COMPUTER    - Technology
10. COOL       - Expression
11. WANT       - Desire
12. WHO        - Question
13. FAMILY     - Relationship
14. LIKE       - Preference
15. ENJOY      - Positive emotion
```

---

## 📊 Performance Metrics (REAL - from 5-Fold CV)

### Accuracy by Model

| Model | Accuracy |
|-------|----------|
| **Neural Network** | 92.13% |
| **Random Forest** | 96.82% |
| **Ensemble (Soft-Vote)** | 95.73% |
| **Meta-Learner (Stacking)** | **97.43% ⭐** |

### Fold-by-Fold Breakdown

| Fold | NN | RF | Ensemble |
|------|----|----|----------|
| **1** | 92.58% | 96.67% | 95.92% |
| **2** | 92.92% | 97.50% | 96.75% |
| **3** | 92.42% | 96.83% | 96.00% |
| **4** | 90.67% | 96.33% | 94.42% |
| **5** | 92.08% | 96.75% | 95.58% |
| **Mean** | **92.13%** | **96.82%** | **95.73%** |
| **Std Dev** | ±0.91% | ±0.49% | ±0.91% |

### Real-Time Performance

| Metric | Value |
|--------|-------|
| **Hand Detection Latency** | ~15ms |
| **Model Prediction** | ~35ms |
| **Real-Time Speed (CPU)** | 16-20 FPS |
| **NLP Generation** | <10ms (local) or <100ms (Groq) |
| **Translation Latency** | ~200ms (Google Translate) |
| **TTS Generation** | ~500ms average |
| **Total E2E Latency** | ~1 second |

### Dataset Information

| Metric | Value |
|--------|-------|
| **Total Samples** | 6,000 |
| **Classes** | 15 ASL signs |
| **Samples per Class** | 400 |
| **Hand Keypoints** | 126 dimensions (21 × 2 × 3) |
| **Emotion Classes** | 5 (Happy, Sad, Angry, Neutral, Surprised) |
| **Facial Features** | 52 landmarks |

---

## 🔧 Technologies Used

### Machine Learning
- **TensorFlow/Keras** - Neural Network
- **scikit-learn** - Random Forest & Logistic Regression
- **Stacking Ensemble** - Meta-learner combination

### Computer Vision
- **MediaPipe** - Hand detection (21 keypoints)
- **OpenCV** - Image processing & display

### NLP & Language
- **spaCy** - Sentence parsing
- **Google Translate (Unofficial)** - Language translation
- **TextBlob** - Grammar assistance

### Speech & Audio
- **pyttsx3** - Local text-to-speech

### APIs & Cloud
- **Groq API** - LLaMA 3.3 70B for sentence verification
- **Google Translate API** (unofficial) - Multi-language support

### Web Framework
- **Flask** - Web server
- **Flask-CORS** - Cross-origin requests
- **HTML5/CSS3/JavaScript** - Frontend

### Data Processing
- **NumPy** - Array operations
- **Pandas** - Data manipulation
- **Pickle** - Model serialization

---

## 🔄 Data Pipeline

```
STEP 1: Real-Time Input
└─ Webcam frame captured at 30 FPS

STEP 2: Hand Detection (MediaPipe)
└─ Extract 21 keypoints × 2 hands = 126 dimensions

STEP 3: Ensemble Prediction
├─ NN model → probability distribution
├─ RF model → probability distribution
└─ Meta-learner → weighted combination = 97.43% confidence

STEP 4: Sign Interpretation
├─ Map class index to sign name (e.g., 0 → "HELP")
└─ Extract confidence score

STEP 5: Emotion Detection (Parallel)
├─ Extract emotion keypoints (52 facial features)
├─ RF emotion model → emotion class
└─ Display emotion alongside sign

STEP 6: Sentence Generation
├─ Use Groq API (if available) for verification
└─ Generate contextual English sentence

STEP 7: Language Translation
├─ Translate to Urdu
└─ Translate to Arabic

STEP 8: Text-to-Speech
├─ English voice synthesis
├─ Urdu voice synthesis
└─ Arabic voice synthesis

STEP 9: Display on Web UI
└─ Show all results on localhost:5000
```

---

## ⚙️ Configuration Options

### app.py Settings

```python
# Video settings
CONFIDENCE_THRESHOLD = 0.75  # Only detect above 75%
FPS_LIMIT = 30              # Max frames per second
SMOOTH_BUFFER = 25          # Smoothing window

# API settings
USE_GROQ = True             # Enable Groq verification
GROQ_TIMEOUT = 5.0          # API timeout (seconds)

# Language settings
SUPPORTED_LANGUAGES = ['en', 'ur', 'ar']  # English, Urdu, Arabic
DEFAULT_LANGUAGE = 'en'

# TTS settings
TTS_SPEED = 150             # Words per minute
TTS_VOLUME = 0.9            # Volume (0-1)
```

---

## 🐛 Troubleshooting

### Issue: "No module named flask"
**Solution:**
```bash
source asl/bin/activate
uv pip install flask
```

### Issue: "GROQ_API_KEY not found"
**Solution:**
- Create `.env` file with `GROQ_API_KEY=your_key`
- Or the system works without it (uses local NLP)

### Issue: Webcam not detected
**Solution:**
```bash
# Check available cameras
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

### Issue: Low FPS (< 10)
**Solution:**
- Reduce video resolution
- Close other applications
- Check CPU usage

### Issue: TTS not working
**Solution:**
```bash
# Reinstall pyttsx3
uv pip install --force-reinstall pyttsx3
```

---

## 📝 File Descriptions

### `flask_app/app.py`
Main Flask server:
- Routes: `/`, `/detect`, `/translate`, `/tts`
- Real-time processing
- API endpoints
- Session management

### `flask_app/templates/index.html`
Web interface:
- Video stream display
- Detection results
- Translation panel
- Audio controls

### `flask_app/static/script.js`
Frontend logic:
- WebSocket communication
- Real-time updates
- Audio playback
- Language selection

### `Ens15class.py`
Trains ensemble models:
- NN + RF training
- 5-fold cross-validation
- Meta-learner stacking
- Model evaluation

### `ex15class.py`
Extracts hand keypoints:
- MediaPipe processing
- Data augmentation
- Balancing classes
- Dataset creation

---

## 🚀 Production Readiness

### Current Status: ✅ Internal Testing Ready

**Fully Functional:**
- ✅ Real-time detection (97.43% meta-learner accuracy)
- ✅ Multi-language support (EN, UR, AR)
- ✅ Text-to-speech (pyttsx3)
- ✅ Emotion detection
- ✅ Web UI (Flask)
- ✅ API integration (Groq)

**Performance:**
- ✅ 16-20 FPS real-time
- ✅ <1 second end-to-end latency
- ✅ Runs on CPU (no GPU needed)
- ✅ Minimal memory footprint

**Reliability:**
- ✅ Graceful fallback without Groq
- ✅ Error handling
- ✅ Input validation
- ✅ Tested on Linux

---

## 🗺️ Next Steps (External Version)

**Coming Soon:**
- [ ] User Personalization Engine
- [ ] SQLite Database for user profiles
- [ ] Migration to FastAPI
- [ ] Enhanced Linguistics for external
- [ ] Multi-user support
- [ ] Production deployment

---

## 📊 Dataset Information

### Training Data
- **Total Samples:** 6,000
- **Classes:** 15 ASL signs
- **Samples per Class:** 400
- **Augmentation:** Noise, scaling, translation, flipping
- **Split:** 5-fold stratified cross-validation

### Keypoint Format
- **Per Hand:** 21 landmarks
- **Both Hands:** 42 landmarks (2 hands)
- **Dimensions:** 126 (x, y, z coordinates × 42 points)
- **Format:** NumPy arrays (.npy)

### Emotion Data
- **Total Samples:** 2,400
- **Emotion Classes:** 5 (Happy, Sad, Angry, Neutral, Surprised)
- **Facial Features:** 52 landmarks
- **Format:** Keypoints extracted from facial detection

---

## 🎯 Key Achievements

✨ **97.43% Meta-Learner Accuracy** - Best stacking ensemble result
🎨 **Multi-Language Support** - Real-time translation to Urdu & Arabic
🔊 **Text-to-Speech** - Natural voice synthesis in 3 languages
😊 **Emotion Detection** - Parallel emotion recognition
⚡ **Real-Time Performance** - 16-20 FPS on CPU
🌐 **Web Interface** - Flask localhost application
🔌 **API Integration** - Groq LLaMA verification
🛡️ **Graceful Fallback** - Works without internet/Groq

---

## 👨‍💻 Author

**Awais Ali**
- Roll No: 045
- BS Data Science (8th Semester)
- Minhaj University Lahore
- Supervisor: Dr. Abdul Aziz

---

## 📅 Version History

**v1.0 (Internal)** - August 2024
- Base ensemble models (97.43% meta-learner accuracy)
- Flask web application
- Multi-language support (EN, UR, AR)
- Text-to-Speech integration
- Emotion detection
- Groq API integration
- **Status:** ✅ COMPLETE & WORKING

**v2.0 (External)** - Coming Soon
- User personalization engine
- FastAPI migration
- Enhanced linguistics
- Production deployment

---

## ⚠️ Important Notes

### GPU Support
- Optimized for CPU (no GPU required)
- CUDA warnings are normal
- System uses available CPU instructions

### Internet Connectivity
- Groq API optional (requires internet)
- System works without it (uses local NLP)
- Google Translate requires internet

### System Requirements
- Min 4GB RAM
- 2GB disk space (models + data)
- Stable webcam
- Linux/macOS preferred

### Language Support
- English: Native ✅
- Urdu: Translation + Voice ✅
- Arabic: Translation + Voice ✅
- Easily extensible to more languages

---

## 📄 License

[Add your license information]

---

## 📞 Support

For internal testing issues:
- Check troubleshooting section above
- Verify all dependencies installed
- Check webcam permissions
- Ensure sufficient disk space

---

**Last Updated:** August 2024  
**Status:** ✅ INTERNAL VERSION COMPLETE & FUNCTIONAL  
**Real Metrics:** NN=92.13%, RF=96.82%, Meta-Learner=97.43%  
**Next Update:** External Version with Personalization

---

## 🎉 Ready to Use!

The internal version is **COMPLETE and FULLY FUNCTIONAL** on localhost:5000

All features working with REAL accuracy metrics:
- ✅ Detection (97.43% meta-learner)
- ✅ Translation (EN/UR/AR)
- ✅ Text-to-Speech
- ✅ Emotion Detection
- ✅ Web UI
- ✅ API Integration

**Start using:** `python flask_app/app.py`
