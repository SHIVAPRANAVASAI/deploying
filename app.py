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
import cv2

st.set_page_config(page_title="AI Data Analysis", page_icon="🧠", layout="wide")
st.title("🧠 Unstructured Data Analysis")

tab1, tab2, tab3 = st.tabs(["🖼️ Image Analysis", "🎧 Audio Analysis", "📝 Text Analysis"])


# ==================== TAB 1: IMAGE ANALYSIS ====================
with tab1:
    
    st.header("🖼️ Advanced Image Analysis")
    st.write("Upload an image for AI-powered face detection, emotion recognition, age/gender prediction, and background removal.")

    # Use OpenCV detector backend (fastest and most reliable)
    detector_backend = "opencv"

    st.markdown("---")

    # -----------------------------
    # Upload Image
    # -----------------------------
    uploaded_image = st.file_uploader(
        "📤 Upload an image", 
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported formats: JPG, JPEG, PNG, WEBP"
    )

    if uploaded_image:
        # Load and display image
        img = Image.open(uploaded_image).convert("RGB")
        
        col_img, col_info = st.columns([1, 2])
        
        with col_img:
            st.image(img, caption="📸 Uploaded Image", use_container_width=True)
        
        with col_info:
            st.info(f"**Image Details:**\n- Format: {img.format}\n- Size: {img.size}\n- Mode: {img.mode}")

        img_array = np.array(img)

        st.markdown("---")
        st.subheader("🔍 Analysis Tools")
        
        # Create 4 columns for analysis buttons
        col1, col2, col3, col4 = st.columns(4)

        # Column 1: Face Detection
        with col1:
            st.markdown("### 👤 Face Detection")
            if st.button("🔎 Detect Face", use_container_width=True, type="primary"):
                with st.spinner("Detecting face..."):
                    try:
                        # Use specified detector backend
                        face_objs = DeepFace.extract_faces(
                            img_path=img_array,
                            detector_backend=detector_backend,
                            enforce_detection=True,
                            align=True
                        )
                        
                        if face_objs:
                            face = face_objs[0]
                            detected_face = (face['face'] * 255).astype(np.uint8)
                            confidence = face['confidence']
                            
                            st.success(f"✅ Face detected! (Confidence: {confidence:.2%})")
                            st.image(detected_face, caption=f"Detected Face", use_container_width=True)
                            
                            # Show facial region info
                            region = face['facial_area']
                            st.caption(f"📍 Position: ({region['x']}, {region['y']}) | Size: {region['w']}x{region['h']}")
                        else:
                            st.warning("⚠️ No face detected in the image.")
                            
                    except Exception as e:
                        st.error(f"❌ Face detection failed: {str(e)}")
                        st.info("💡 Try using a different detector model or ensure the image contains a clear face.")

        # Column 2: Age & Gender Detection
        with col2:
            st.markdown("### 👥 Age & Gender")
            if st.button("🎯 Analyze Demographics", use_container_width=True, type="primary"):
                with st.spinner("Analyzing age and gender..."):
                    try:
                        analysis = DeepFace.analyze(
                            img_path=img_array,
                            actions=['age', 'gender'],
                            detector_backend=detector_backend,
                            enforce_detection=True,
                            silent=True
                        )
                        
                        result = analysis[0]
                        predicted_age = result['age']
                        gender_data = result['gender']
                        dominant_gender = result['dominant_gender']
                        
                        st.success("✅ Analysis complete!")
                        
                        # Display results with metrics
                        st.metric("Predicted Age", f"{predicted_age} years")
                        st.metric("Gender", dominant_gender.capitalize())
                        
                        # Show confidence breakdown
                        st.caption("**Gender Confidence:**")
                        st.caption(f"👨 Male: {gender_data['Man']:.1f}%")
                        st.caption(f"👩 Female: {gender_data['Woman']:.1f}%")
                        
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                        st.info("💡 Make sure the face is clearly visible and well-lit.")

        # Column 3: Emotion Detection
        with col3:
            st.markdown("### 😊 Emotion")
            if st.button("🎭 Detect Emotion", use_container_width=True, type="primary"):
                with st.spinner("Detecting emotion..."):
                    try:
                        analysis = DeepFace.analyze(
                            img_path=img_array,
                            actions=['emotion'],
                            detector_backend=detector_backend,
                            enforce_detection=True,
                            silent=True
                        )
                        
                        result = analysis[0]
                        emotions = result['emotion']
                        dominant_emotion = result['dominant_emotion']
                        
                        # Emoji mapping
                        emotion_emoji = {
                            'angry': '😠', 'disgust': '🤢', 'fear': '😨',
                            'happy': '😊', 'sad': '😢', 'surprise': '😲',
                            'neutral': '😐'
                        }
                        
                        st.success("✅ Emotion detected!")
                        st.markdown(f"### {emotion_emoji.get(dominant_emotion, '😐')} {dominant_emotion.capitalize()}")
                        
                        # Show all emotions with progress bars
                        st.caption("**Emotion Breakdown:**")
                        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
                        for emotion, score in sorted_emotions[:3]:
                            st.progress(score / 100)
                            st.caption(f"{emotion_emoji.get(emotion, '•')} {emotion.capitalize()}: {score:.1f}%")
                        
                    except Exception as e:
                        st.error(f"❌ Emotion detection failed: {str(e)}")
                        st.info("💡 Ensure the face has a clear expression visible.")

        # Column 4: Background Removal
        with col4:
            st.markdown("### 🎨 Background Removal")
            if st.button("✂️ Remove Background", use_container_width=True, type="primary"):
                with st.spinner("Removing background..."):
                    try:
                        # Apply background removal
                        output_image = remove(img)
                        
                        st.success("✅ Background removed!")
                        st.image(output_image, caption="Background Removed", use_container_width=True)
                        
                        # Offer download
                        img_byte_arr = io.BytesIO()
                        output_image.save(img_byte_arr, format='PNG')
                        img_byte_arr = img_byte_arr.getvalue()
                        
                        st.download_button(
                            label="📥 Download PNG",
                            data=img_byte_arr,
                            file_name="background_removed.png",
                            mime="image/png",
                            use_container_width=True
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Background removal failed: {str(e)}")
                        st.info("💡 Try a different image or check internet connection.")

        # Advanced: Full Analysis Button
        st.markdown("---")
        if st.button("🚀 Run Complete Analysis (All Features)", use_container_width=True):
            with st.spinner("Running comprehensive analysis..."):
                try:
                    analysis = DeepFace.analyze(
                        img_path=img_array,
                        actions=['age', 'gender', 'emotion', 'race'],
                        detector_backend=detector_backend,
                        enforce_detection=True,
                        silent=True
                    )
                    
                    result = analysis[0]
                    
                    st.success("✅ Complete analysis finished!")
                    
                    # Display comprehensive results
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.markdown("#### 📊 Demographics")
                        st.write(f"**Age:** {result['age']} years")
                        st.write(f"**Gender:** {result['dominant_gender'].capitalize()}")
                        st.write(f"**Ethnicity:** {result['dominant_race'].capitalize()}")
                    
                    with col_b:
                        st.markdown("#### 😊 Emotion")
                        st.write(f"**Dominant:** {result['dominant_emotion'].capitalize()}")
                        st.write("**Top 3 Emotions:**")
                        sorted_emotions = sorted(result['emotion'].items(), key=lambda x: x[1], reverse=True)
                        for emotion, score in sorted_emotions[:3]:
                            st.write(f"- {emotion.capitalize()}: {score:.1f}%")
                    
                    with col_c:
                        st.markdown("#### 👤 Face Region")
                        region = result['region']
                        st.write(f"**Position:** ({region['x']}, {region['y']})")
                        st.write(f"**Size:** {region['w']} x {region['h']}px")
                        st.write(f"**Detector:** {detector_backend}")
                    
                except Exception as e:
                    st.error(f"❌ Complete analysis failed: {str(e)}")

    else:
        st.info("👆 Upload an image to get started with AI-powered analysis!")
        st.markdown("""
        **Supported Features:**
        - 👤 Face Detection with multiple models
        - 👥 Age & Gender Prediction
        - 😊 Emotion Recognition
        - 🎨 Background Removal
        - 🔍 Full Demographic Analysis
        """)


# ==================== TAB 2: AUDIO ANALYSIS ====================




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



