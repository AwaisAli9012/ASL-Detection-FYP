# ASL Sign Language Detection System

A real-time American Sign Language (ASL) detection system built for a Final Year Project (FYP). The system uses computer vision and deep learning to recognize ASL hand signs and convert them into natural English sentences using AI.

## Overview

The system detects hand landmarks in real time using a webcam, classifies them into ASL signs, builds sentences from the signed words, and uses the Groq AI API to generate two natural English interpretations — one from the signer's perspective and one as an instruction.

## Features

- Real-time ASL sign detection using webcam
- 15 ASL word classes supported
- Hand landmark extraction using MediaPipe
- Deep learning model trained on WLASL dataset
- Dual sentence interpretation using Groq AI (Self and Instruction context)
- Sentence builder with keyboard controls
- Both hand orientations supported (left and right hand)

## Dataset

This project uses the **WLASL (World Level American Sign Language)** dataset — a professional dataset containing thousands of ASL sign videos recorded by real ASL signers.

- 15 classes selected from 100 extracted classes
- Hand detection filtering using MediaPipe to remove non-hand frames
- Data augmentation including horizontal flipping for both hand orientations
- Balanced augmentation applied — 300 samples per class

## Supported Signs

| # | Sign | # | Sign | # | Sign |
|---|------|---|------|---|------|
| 1 | help | 6 | finish | 11 | want |
| 2 | yes | 7 | play | 12 | who |
| 3 | no | 8 | mother | 13 | family |
| 4 | before | 9 | computer | 14 | like |
| 5 | go | 10 | cool | 15 | enjoy |

## System Architecture
Webcam Input
↓
MediaPipe Hand Detection
↓
Landmark Extraction (126 keypoints per frame)
↓
Neural Network Classifier (15 classes)
↓
Sentence Builder
↓
Groq AI Sentence Generation
↓
Dual Interpretation Display

## Model Details

- **Architecture:** Fully connected neural network
- **Input:** 126 hand landmark coordinates (21 points × 3 axes × 2 hands)
- **Layers:** Dense(512) → Dense(256) → Dense(128) → Dense(64) → Dense(15)
- **Training samples:** 4500 (balanced augmentation, 300 per class)
- **Validation accuracy:** ~83%
- **Training:** EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

## Project Structure
NEW/
├── 10_classes/
│   ├── ex10class.py       # Keypoint extraction for 10 classes
│   ├── trn10class.py      # Training script for 10 classes
│   └── det10class.py      # Detection script for 10 classes
├── 15_classes/
│   ├── ex15class.py       # Keypoint extraction for 15 classes
│   ├── trn15class.py      # Training script for 15 classes
│   └── det15class.py      # Detection script for 15 classes
└── README.md

## Requirements
tensorflow
mediapipe
opencv-python
numpy
scikit-learn
groq

Install all dependencies:
```bash
pip install tensorflow mediapipe opencv-python numpy scikit-learn groq
```

## Setup and Usage

**1. Clone the repository**
```bash
git clone https://github.com/AwaisAli9012/ASL-Detection-FYP.git
cd ASL-Detection-FYP
```

**2. Install dependencies**
```bash
pip install tensorflow mediapipe opencv-python numpy scikit-learn groq
```

**3. Set your Groq API key**
```bash
set GROQ_API_KEY=your_groq_api_key_here
```
Get a free API key from [console.groq.com](https://console.groq.com)

**4. Run detection**
```bash
cd 15_classes
python det15class.py
```

## Controls

| Key | Action |
|-----|--------|
| ENTER | Add current detected word to sentence |
| BACKSPACE | Remove last word from sentence |
| G | Generate AI sentence interpretations |
| SPACE | Clear everything |
| Q | Quit |

## How It Works

1. Webcam captures live video
2. MediaPipe detects hand landmarks in each frame
3. 126 coordinates extracted per frame (21 landmarks × XYZ × 2 hands)
4. Neural network classifies the hand shape into one of 15 ASL signs
5. User presses ENTER to confirm each word
6. User presses G to send signed words to Groq AI
7. Groq generates two interpretations displayed on screen

## Results

- 15 class model achieves ~83% validation accuracy
- Real-time detection at smooth frame rate
- Both left and right hand signing supported
- Balanced dataset with 300 samples per class
- Groq AI generates two natural English sentence interpretations

## Acknowledgements

- [WLASL Dataset](https://github.com/dxli94/WLASL) by Dongxu Li et al.
- [MediaPipe](https://mediapipe.dev/) by Google
- [Groq API](https://console.groq.com) for AI sentence generation 
## Dataset Download 
 
The preprocessed dataset used in this project is available on Kaggle: 
 
[Download Dataset](https://www.kaggle.com/datasets/awaisali9012/sign-language-dataset) 
