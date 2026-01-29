# 🛡️ Dial-Defenders  
### AI-Powered Voice Authenticity Detection for Scam & Fraud Prevention

> Built for **HCL GUVI India AI Impact Buildathon 2026**  
> Designed with focus on **User Safety, Explainability, and Real-World Deployment**

---

## 🚨 Problem Statement

AI-generated voices are increasingly being misused for:
- Scam and fraud calls
- Impersonation of trusted individuals
- Fake emergency situations
- Financial and identity theft

These voices are highly realistic and **work across languages**, making
traditional keyword-based or rule-based detection systems ineffective.

**Dial-Defenders** aims to solve this problem by identifying whether a given
voice sample is **AI-generated or human**, based purely on **acoustic behaviour**.

---

## 💡 Our Key Insight (What Makes This Solution Different)

Most existing systems focus on:
- *What* is being said (speech-to-text analysis)  
❌ Language dependent  
❌ Easily bypassed by script changes  

Our approach focuses on:
- **How the voice behaves acoustically**

### 🔍 Core Observation:
AI-generated voices exhibit:
- Unnaturally uniform spectral patterns
- Reduced micro-variations
- Lower organic jitter compared to human speech

Human voices naturally contain imperfections that AI-generated audio often lacks.

This makes our system:
- ✅ Language-independent  
- ✅ Robust against scripted manipulation  
- ✅ Suitable for real-world scam detection  

---

## 🧠 Technical Approach

### 🎧 Input
- Base64-encoded MP3 audio
- Supports **Tamil, Hindi, English, Malayalam, Telugu**
- No dependency on transcription or keywords

### ⚙️ Processing Pipeline
1. Decode Base64 audio input
2. Load audio waveform using **Librosa**
3. Extract acoustic features
4. Analyze **Spectral Flatness**
5. Classify voice as:
   - `AI_GENERATED`
   - `HUMAN`

### 📤 Output (Explainable JSON Response)

```json
{
  "classification": "AI_GENERATED",
  "confidence": 0.92,
  "explanation": "Detected robotic spectral uniformity in high-frequency bands."
}
---

## 🧪 Prototype Status

> **Note:**  
> The current prototype focuses on **signal-based acoustic analysis** to identify  
> AI-generated voice patterns such as **spectral uniformity** and **reduced vocal variance**.  
>  
> This approach ensures **language independence**, **real-time performance**,  
> and **low computational overhead** at the prototype stage.  
>  
> **ML-based classification will be integrated in subsequent iterations** to  
> enhance accuracy, adaptability across diverse datasets, and robustness against  
> evolving AI voice synthesis techniques.

---

## 🔮 Future Scope & Roadmap

Planned enhancements for **Dial-Defenders** include:
- Integration of lightweight ML-based classifiers  
- Training on multi-language and multi-accent datasets  
- Improved confidence calibration and scoring mechanisms  
- Robust detection against advanced AI voice cloning techniques  
- Optimization for edge and low-resource devices  
- Real-time API integration with call screening and fraud detection systems  

---

## 🏆 Impact & Vision

Dial-Defenders addresses a **real and rapidly growing threat** in digital safety.  
By focusing on **acoustic behaviour instead of content**, the solution remains  
**scalable, explainable, and resilient** across languages and real-world use cases.  

This project lays a strong foundation for **future-ready, AI-driven voice fraud  
prevention systems** with practical deployment potential.