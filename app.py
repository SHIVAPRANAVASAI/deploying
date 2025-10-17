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

from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import random

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
                            enforce_detection=False,  # Faster - doesn't re-detect face
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
                            enforce_detection=False,  # Faster - doesn't re-detect face
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
                        enforce_detection=False,  # Faster - doesn't re-detect face
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


# ==================== TAB 3: TEXT ANALYSIS ====================
with tab3:
    st.header("📝 Text Analysis with NLP")
    st.write("Analyze text to extract parts of speech and visualize word clouds.")

    # Hardcoded sample stories
    stories = [
        """In a remote kingdom nestled between jagged mountains and endless forests, Princess Elara spent her days exploring the sprawling royal gardens, which stretched beyond what the eye could see. Ancient trees, taller than any castle spire, whispered secrets of centuries past. Streams glimmered like silver ribbons, and hidden nooks revealed forgotten statues of long-lost rulers. Each evening, the princess wandered alone, following the soft glow of luminescent flowers that seemed to respond to her presence. She discovered a small, crystal-clear pond where the water shimmered with reflections of constellations that weren't visible in the night sky above. Mystical creatures, some with wings of iridescent silk and others with scales that reflected the sunlight, emerged from the undergrowth. They spoke in melodious tones, sharing wisdom about magic, courage, and the legacy of her ancestors. Elara meticulously recorded everything in her leather-bound journal, eager to preserve these secrets.""",
        
        """During the bustling era of the 1920s, in a city that never slept, Detective Samuel Hart navigated the labyrinthine streets of New York. The roar of jazz music spilled from dimly lit speakeasies, blending with the clatter of streetcars and the occasional wail of a distant siren. Samuel's office, cluttered with case files and newspapers, smelled of tobacco and ink. Each case presented new challenges: jewel thefts orchestrated with surgical precision, clandestine meetings under the flickering glow of lampposts, and coded messages that tested his intellect. Night after night, he roamed the city in a long trench coat and fedora, cigarette smoke curling around his face as he pieced together clues overlooked by ordinary citizens.""",
        
        """On a distant exoplanet, where the sky shimmered in surreal hues of emerald and violet, Captain Rhea led a team of explorers through canyons carved by ancient rivers of liquid crystal. Each step on the iridescent terrain revealed new flora and fauna, alien yet strangely familiar, pulsating in rhythm with the wind. Towering spires of stone rose like jagged fingers, and the explorers' suits glimmered with embedded sensors that captured every subtle vibration. Bioluminescent plants reacted to human presence, illuminating pathways that seemed almost deliberately designed.""",
        
        """In the neon-lit heart of Tokyo, young coder Akira toiled over lines of code that promised to revolutionize urban transportation. Streets pulsed with energy as neon advertisements flickered over crowded crosswalks, and the hum of trains beneath the city created a constant rhythm. Akira's AI was designed to predict traffic congestion and prevent accidents, analyzing millions of data points in real-time. His workspace was cluttered with multiple monitors, coffee cups, and mechanical keyboards, each keystroke echoing determination and fatigue.""",
        
        """Deep in the Amazon rainforest, a team of scientists embarked on an unprecedented expedition to discover rare medicinal plants. Guided by indigenous elders with knowledge passed down through generations, they navigated treacherous rivers teeming with exotic wildlife, thick canopies that blocked sunlight, and terrain that shifted unpredictably with every step. Nights were spent around flickering campfires, where the symphony of nocturnal creatures filled the air with haunting melodies."""
    ]

    # Initialize session_state for text area
    if "text_area" not in st.session_state:
        st.session_state.text_area = ""

    # Random story button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    with col_btn1:
        if st.button("🎲 Random Story", use_container_width=True):
            st.session_state.text_area = random.choice(stories)
    
    with col_btn2:
        if st.button("🗑️ Clear Text", use_container_width=True):
            st.session_state.text_area = ""

    # Text area (shows random story if chosen, else empty / user input)
    st.session_state.text_area = st.text_area(
        "Paste or modify your text here:", 
        value=st.session_state.text_area, 
        height=250,
        placeholder="Enter or paste your text here, or click 'Random Story' for a sample..."
    )

    # Analyze button
    if st.button("🚀 Analyze Text", use_container_width=True, type="primary"):
        text = st.session_state.text_area.strip()

        if text:
            with st.spinner("Analyzing text..."):
                try:
                    blob = TextBlob(text)
                    words_and_tags = blob.tags  # (word, POS tag)

                    # POS extraction
                    nouns = [word for word, tag in words_and_tags if tag.startswith('NN')]
                    verbs = [word for word, tag in words_and_tags if tag.startswith('VB')]
                    adjectives = [word for word, tag in words_and_tags if tag.startswith('JJ')]
                    adverbs = [word for word, tag in words_and_tags if tag.startswith('RB')]

                    # WordCloud generator
                    def make_wordcloud(words, color):
                        if not words or len(words) == 0:
                            st.warning("No words found for this category.")
                            return None
                        text_for_wc = " ".join(words)
                        wc = WordCloud(
                            width=500, 
                            height=400, 
                            background_color='black', 
                            colormap=color
                        ).generate(text_for_wc)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis("off")
                        return fig

                    st.success("✅ Analysis complete!")
                    st.markdown("---")

                    # Layout 2x2 for word clouds
                    st.subheader("☁️ Word Clouds by Parts of Speech")
                    col1, col2 = st.columns(2)
                    col3, col4 = st.columns(2)

                    with col1:
                        st.markdown("### 🧠 Nouns")
                        fig = make_wordcloud(nouns, "plasma")
                        if fig: st.pyplot(fig)
                    
                    with col2:
                        st.markdown("### ⚡ Verbs")
                        fig = make_wordcloud(verbs, "inferno")
                        if fig: st.pyplot(fig)
                    
                    with col3:
                        st.markdown("### 🎨 Adjectives")
                        fig = make_wordcloud(adjectives, "cool")
                        if fig: st.pyplot(fig)
                    
                    with col4:
                        st.markdown("### 💨 Adverbs")
                        fig = make_wordcloud(adverbs, "magma")
                        if fig: st.pyplot(fig)

                    # Quick stats
                    st.markdown("---")
                    st.subheader("📊 Parts of Speech Statistics")
                    
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    col_stat1.metric("Nouns", len(nouns))
                    col_stat2.metric("Verbs", len(verbs))
                    col_stat3.metric("Adjectives", len(adjectives))
                    col_stat4.metric("Adverbs", len(adverbs))
                    
                    # Additional text statistics
                    st.markdown("---")
                    st.subheader("📈 Text Statistics")
                    col_text1, col_text2, col_text3 = st.columns(3)
                    
                    sentences = blob.sentences
                    words = blob.words
                    
                    col_text1.metric("Total Words", len(words))
                    col_text2.metric("Total Sentences", len(sentences))
                    col_text3.metric("Avg Words/Sentence", f"{len(words) / len(sentences) if len(sentences) > 0 else 0:.1f}")
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.info("💡 Make sure the text is in English and properly formatted.")

        else:
            st.warning("⚠️ Please paste or select some text first.")
    
    else:
        st.info("👆 Enter your text and click 'Analyze Text' to see word clouds and statistics!")


