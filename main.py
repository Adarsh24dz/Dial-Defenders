@app.post("/detect")
async def detect_voice(file: UploadFile = File(...)):
    contents = await file.read()
    audio = AudioSegment.from_file(io.BytesIO(contents))
    audio = audio.set_frame_rate(16000).set_channels(1)
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    y, sr = librosa.load(wav_io, sr=16000)
    
    # Silent check first
    if np.mean(np.abs(y)) < 0.001:  # Very low energy
        return {
            "status": "error",
            "language": "UNKNOWN",
            "classification": "SILENCE",
            "confidenceScore": 1.0,
            "explanation": "No audible voice detected - audio is silent or empty"
        }
    
    # Features
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    pitch_var = np.std(librosa.yin(y, fmin=50, fmax=500, sr=sr))
    energy_var = np.std(librosa.feature.rms(y=y))
    
    ai_score = 0
    if zcr < 0.08: ai_score += 0.2
    if centroid < 1800: ai_score += 0.2  
    if pitch_var < 10: ai_score += 0.2
    if energy_var < 0.01: ai_score += 0.2
    if rolloff < 3500: ai_score += 0.2
    
    classification = "AI_GENERATED" if ai_score > 0.5 else "HUMAN"
    confidence = round(min(ai_score * 2, 1.0), 2) if classification == "AI_GENERATED" else round((1-ai_score) * 2, 2)
    
    # Proper sentence explanations
    if classification == "AI_GENERATED":
        explanation = "Unnatural pitch consistency and robotic speech patterns detected"
    else:
        explanation = "Natural prosody and human-like variations observed"
    
    return {
        "status": "success",
        "language": "TAMIL",
        "classification": classification,
        "confidenceScore": confidence,
        "explanation": explanation
    }