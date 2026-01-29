# Dial-Defenders
AI-Powered Voice Authenticity Detection for Scam & Fraud Prevention

Built for HCL GUVI India AI Impact Buildathon 2026  
Focus: User Safety, Explainability, Real-World Deployment

---

## Problem Statement

AI-generated voices are being misused for scam calls, impersonation,
fake emergencies, and financial fraud.

These voices work across languages and sound realistic,
making keyword-based detection ineffective.

Dial-Defenders detects whether a voice is AI-generated or human
by analysing acoustic behaviour instead of spoken content.

---

## Prototype Status

The current prototype uses signal-based acoustic analysis to detect
AI-generated voice patterns such as spectral uniformity and reduced
vocal variance.

This approach is language-independent, lightweight, and suitable
for real-time use at the prototype stage.

ML-based classification will be added in future iterations
to improve accuracy and robustness.

---

## Technical Overview

Input:
- Base64-encoded MP3 audio
- Supports Tamil, Hindi, English, Malayalam, Telugu

Processing:
1. Decode audio
2. Extract acoustic features using Librosa
3. Analyse spectral flatness
4. Classify voice as AI-generated or human

Output:
```json
{
  "classification": "AI_GENERATED",
  "confidence": 0.92,
  "explanation": "Detected robotic spectral uniformity in high-frequency bands."
}