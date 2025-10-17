import streamlit as st
from gtts import gTTS
import os
import speech_recognition as sr
from pydub import AudioSegment
import io

from rembg import remove
from deepface import DeepFace
import numpy as np
from PIL import Image

st.title("🧠 Unstructured Data Analysis")

tab1, tab2, tab3 = st.tabs(["🖼️ Image Analysis", "🎧 Audio Analysis", "📝 Text Analysis"])



with tab1:
    
    st.title("🖼️ Image Analysis with DeepFace")

    # -----------------------------
    # Upload Image
    # -----------------------------
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image:

        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, caption="Uploaded Image", width=300)

        img_array = np.array(img)

        col1, col2, col3, col4 = st.columns(4)


        with col1:
            if st.button("Detect Face"):
                try:
                    detection = DeepFace.detectFace(img_array, enforce_detection=True)
                    st.success("✅ Face detected!")
                    st.image(detection, caption="Detected Face", use_column_width=True)
                except Exception as e:
                    st.error(f"Face detection failed: {e}")

        with col2:
            if st.button("Detect Age & Gender"):
                try:
                    analysis = DeepFace.analyze(img_path=np.array(img_array), actions=['age', 'gender'], enforce_detection=True)
                    predicted_age = analysis[0]['age']
                    predicted_gender = analysis[0]['dominant_gender']
                    st.success("✅ Age & Gender detected!")
                    st.write(f"**Predicted Age:** {predicted_age}")
                    st.write(f"**Predicted Gender:** {predicted_gender}")
                except Exception as e:
                    st.error(f"Age/Gender detection failed: {e}")

        with col3:
            if st.button("Detect Emotion"):
                try:
                    analysis = DeepFace.analyze(img_path=np.array(img_array), actions=['emotion'], enforce_detection=True)
                    predicted_emotion = analysis[0]['dominant_emotion']
                    st.success("✅ Emotion detected!")
                    st.write(f"**Predicted Emotion:** {predicted_emotion}")
                except Exception as e:
                    st.error(f"Emotion detection failed: {e}")

        with col4:
            output_image = remove(img)
            st.image(output_image, caption="BG Removed Image", width=300)









with tab2:

    # ------------------ TEXT TO SPEECH ------------------
    st.header("🗣️ Text to Speech")
    text = st.text_area("Enter text to convert to speech:")

    if st.button("Convert to Audio"):
        if text.strip():
            tts = gTTS(text, lang='en')
            tts.save("output.mp3")
            audio_file = open("output.mp3", "rb")
            st.audio(audio_file.read(), format='audio/mp3')
            st.success("✅ Conversion complete!")
        else:
            st.warning("Please enter some text.")

    
    # ------------------ SPEECH TO TEXT ------------------
    st.header("🗣️ Speech to Text")

    # Upload audio
    uploaded_audio = st.file_uploader("Upload audio file (wav, mp3, m4a)", type=["wav","mp3","m4a"])

    if uploaded_audio:
        # Convert uploaded audio to PCM WAV
        audio_bytes = uploaded_audio.read()
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        # Play audio in Streamlit
        st.audio(wav_io, format="audio/wav")

        if st.button("Transcribe Audio"):
            recognizer = sr.Recognizer()
            # SpeechRecognition requires a real file-like object, so we reset BytesIO
            wav_io.seek(0)
            with sr.AudioFile(wav_io) as source:
                audio_data = recognizer.record(source)

            with st.spinner("Transcribing..."):
                try:
                    text_output = recognizer.recognize_google(audio_data)
                    st.success("✅ Transcription complete!")
                    st.subheader("Transcribed Text")
                    st.write(text_output)
                except sr.UnknownValueError:
                    st.error("Speech not recognized.")
                except sr.RequestError:
                    st.error("Google API unavailable or network error.")



