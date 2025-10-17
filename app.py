import os

# Set environment variables FIRST, before any other imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_USE_LEGACY_KERAS'] = '1'  # Force Keras 2 compatibility mode

import streamlit as st
from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment
import io

import numpy as np
from PIL import Image
import cv2

from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import random

st.set_page_config(page_title="AI Data Analysis", page_icon="🧠", layout="wide")
st.title("🧠 Unstructured Data Analysis")

# Lazy loading functions with caching
@st.cache_resource
def load_deepface():
    """Lazy load DeepFace only when needed"""
    from deepface import DeepFace
    return DeepFace

@st.cache_resource
def load_rembg():
    """Lazy load rembg only when needed"""
    from rembg import remove
    return remove

@st.cache_resource
def load_spacy():
    """Lazy load spacy only when needed"""
    import spacy
    from spacy import displacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.error("❌ Spacy model not found. Please wait while it downloads...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    return nlp, displacy

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
            st.image(img, caption="📸 Uploaded Image", width='stretch')
        
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
            if st.button("🔎 Detect Face", key="detect_face", type="primary"):
                with st.spinner("Loading AI model..."):
                    DeepFace = load_deepface()
                with st.spinner("Detecting face..."):
                    try:
                        # Use specified detector backend
                        face_objs = DeepFace.extract_faces(
                            img_path=img_array,
                            detector_backend=detector_backend,
                            enforce_detection=False,
                            align=True
                        )
                        
                        if face_objs:
                            face = face_objs[0]
                            detected_face = (face['face'] * 255).astype(np.uint8)
                            confidence = face['confidence']
                            
                            st.success(f"✅ Face detected! (Confidence: {confidence:.2%})")
                            st.image(detected_face, caption=f"Detected Face", width='stretch')
                            
                            # Show facial region info
                            region = face['facial_area']
                            st.caption(f"📍 Position: ({region['x']}, {region['y']}) | Size: {region['w']}x{region['h']}")
                        else:
                            st.warning("⚠️ No face detected in the image.")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

        # Column 2: Demographics Analysis
        with col2:
            st.markdown("### 👥 Demographics")
            if st.button("🎯 Analyze Demographics", key="analyze_demo", type="primary"):
                with st.spinner("Loading AI model..."):
                    DeepFace = load_deepface()
                with st.spinner("Analyzing demographics..."):
                    try:
                        result = DeepFace.analyze(
                            img_path=img_array,
                            actions=['age', 'gender'],
                            detector_backend=detector_backend,
                            enforce_detection=False
                        )
                        
                        if isinstance(result, list):
                            result = result[0]
                        
                        age = result['age']
                        gender = result['dominant_gender']
                        gender_conf = result['gender'][gender]
                        
                        st.success("✅ Analysis complete!")
                        
                        # Display results
                        st.metric("Age", f"{age} years")
                        st.metric("Gender", f"{gender.title()} ({gender_conf:.1f}%)")
                        
                        # Gender distribution
                        st.caption("**Gender Confidence:**")
                        for g, conf in result['gender'].items():
                            st.progress(conf/100, text=f"{g.title()}: {conf:.1f}%")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

        # Column 3: Emotion Detection
        with col3:
            st.markdown("### 🎭 Emotion")
            if st.button("🎭 Detect Emotion", key="detect_emotion", type="primary"):
                with st.spinner("Loading AI model..."):
                    DeepFace = load_deepface()
                with st.spinner("Analyzing emotion..."):
                    try:
                        result = DeepFace.analyze(
                            img_path=img_array,
                            actions=['emotion'],
                            detector_backend=detector_backend,
                            enforce_detection=False
                        )
                        
                        if isinstance(result, list):
                            result = result[0]
                        
                        emotion = result['dominant_emotion']
                        emotion_conf = result['emotion'][emotion]
                        
                        # Emotion emoji mapping
                        emotion_emojis = {
                            'happy': '😄', 'sad': '😢', 'angry': '😠',
                            'surprise': '😲', 'fear': '😨', 'disgust': '🤢',
                            'neutral': '😐'
                        }
                        
                        emoji = emotion_emojis.get(emotion, '😐')
                        st.success(f"✅ {emoji} {emotion.title()}")
                        st.metric("Confidence", f"{emotion_conf:.1f}%")
                        
                        # Emotion distribution
                        st.caption("**All Emotions:**")
                        sorted_emotions = sorted(result['emotion'].items(), key=lambda x: x[1], reverse=True)
                        for em, conf in sorted_emotions:
                            st.progress(conf/100, text=f"{em.title()}: {conf:.1f}%")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

        # Column 4: Background Removal
        with col4:
            st.markdown("### ✂️ Background")
            if st.button("✂️ Remove Background", key="remove_bg", type="primary"):
                with st.spinner("Loading AI model..."):
                    remove = load_rembg()
                with st.spinner("Removing background..."):
                    try:
                        output_image = remove(img)
                        
                        st.success("✅ Background removed!")
                        st.image(output_image, caption="Background Removed", width='stretch')
                        
                        # Download button
                        buf = io.BytesIO()
                        output_image.save(buf, format="PNG")
                        st.download_button(
                            label="📥 Download PNG",
                            data=buf.getvalue(),
                            file_name="background_removed.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

        st.markdown("---")
        
        # Complete Analysis Button
        if st.button("🚀 Run Complete Analysis (All Features)", key="complete_analysis"):
            with st.spinner("Loading AI models..."):
                DeepFace = load_deepface()
                remove = load_rembg()
            
            with st.spinner("Running complete analysis... This may take a moment."):
                try:
                    # Run all analyses
                    result = DeepFace.analyze(
                        img_path=img_array,
                        actions=['age', 'gender', 'emotion'],
                        detector_backend=detector_backend,
                        enforce_detection=False
                    )
                    
                    if isinstance(result, list):
                        result = result[0]
                    
                    # Create columns for results
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.subheader("👥 Demographics")
                        st.metric("Age", f"{result['age']} years")
                        gender = result['dominant_gender']
                        st.metric("Gender", f"{gender.title()} ({result['gender'][gender]:.1f}%)")
                    
                    with col_b:
                        st.subheader("🎭 Emotion")
                        emotion = result['dominant_emotion']
                        emotion_emojis = {
                            'happy': '😄', 'sad': '😢', 'angry': '😠',
                            'surprise': '😲', 'fear': '😨', 'disgust': '🤢',
                            'neutral': '😐'
                        }
                        emoji = emotion_emojis.get(emotion, '😐')
                        st.metric("Emotion", f"{emoji} {emotion.title()}")
                        st.metric("Confidence", f"{result['emotion'][emotion]:.1f}%")
                    
                    with col_c:
                        st.subheader("✂️ Background Removal")
                        with st.spinner("Processing..."):
                            output_image = remove(img)
                            st.image(output_image, caption="No Background", width='content')
                    
                    st.success("✅ Complete analysis finished!")
                
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")

# ==================== TAB 2: AUDIO ANALYSIS ====================
with tab2:
    st.header("🎧 Audio Analysis & Processing")
    
    st.subheader("🔊 Text to Speech")
    text_input = st.text_area("Enter text to convert to speech:", 
                              "Hello! Welcome to AI Data Analysis.",
                              height=100)
    
    lang_option = st.selectbox("Select Language:", 
                               ["en", "es", "fr", "de", "hi", "ja", "zh-CN"])
    
    if st.button("🎵 Generate Speech", type="primary"):
        if text_input:
            with st.spinner("Generating speech..."):
                try:
                    tts = gTTS(text=text_input, lang=lang_option, slow=False)
                    audio_bytes = io.BytesIO()
                    tts.write_to_fp(audio_bytes)
                    audio_bytes.seek(0)
                    
                    st.success("✅ Speech generated!")
                    st.audio(audio_bytes, format="audio/mp3")
                    
                    st.download_button(
                        label="📥 Download Audio",
                        data=audio_bytes.getvalue(),
                        file_name="speech.mp3",
                        mime="audio/mp3"
                    )
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please enter some text first.")
    
    st.markdown("---")
    st.subheader("🎤 Speech to Text")
    st.info("📌 **Note:** Upload an audio file (WAV format works best) for transcription.")
    
    uploaded_audio = st.file_uploader("📤 Upload Audio File", 
                                     type=["wav", "mp3", "ogg", "flac"])
    
    if uploaded_audio:
        st.audio(uploaded_audio, format=f"audio/{uploaded_audio.type.split('/')[-1]}")
        
        if st.button("🎯 Transcribe Audio", type="primary"):
            with st.spinner("Transcribing audio..."):
                try:
                    # Save uploaded file temporarily
                    audio_bytes = uploaded_audio.read()
                    
                    # Convert to WAV if needed
                    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
                    wav_io = io.BytesIO()
                    audio.export(wav_io, format="wav")
                    wav_io.seek(0)
                    
                    # Transcribe
                    recognizer = sr.Recognizer()
                    with sr.AudioFile(wav_io) as source:
                        audio_data = recognizer.record(source)
                        text = recognizer.recognize_google(audio_data)
                    
                    st.success("✅ Transcription complete!")
                    st.text_area("📝 Transcribed Text:", text, height=150)
                    
                except sr.UnknownValueError:
                    st.error("❌ Could not understand the audio. Please try with a clearer recording.")
                except sr.RequestError as e:
                    st.error(f"❌ Could not request results: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# ==================== TAB 3: TEXT ANALYSIS ====================
with tab3:
    st.header("📝 Advanced Text Analysis")
    
    # Sample stories for random generation
    sample_stories = [
        "The quick brown fox jumps over the lazy dog near the beautiful park in New York.",
        "Apple Inc. announced a new product launch in California next month.",
        "Shakespeare wrote many famous plays in London during the Renaissance period.",
        "NASA successfully launched a mission to Mars from Kennedy Space Center.",
        "The Amazon rainforest is home to countless species and spans across South America."
    ]
    
    # Text input with sample button
    col_text, col_buttons = st.columns([3, 1])
    
    with col_text:
        text = st.text_area("Enter text for analysis:", 
                           placeholder="Type or paste your text here...",
                           height=150,
                           key="text_input")
    
    with col_buttons:
        if st.button("🎲 Random Story"):
            st.session_state.text_input = random.choice(sample_stories)
            st.rerun()
        if st.button("🗑️ Clear Text"):
            st.session_state.text_input = ""
            st.rerun()
    
    st.markdown("---")
    
    # Analysis options
    analysis_options = st.multiselect(
        "Select Analysis Types:",
        ["Sentiment Analysis", "Word Cloud", "Named Entity Recognition (NER)", "Text Statistics"],
        default=["Sentiment Analysis"]
    )
    
    if st.button("🚀 Analyze Text", type="primary"):
        if not text:
            st.warning("⚠️ Please enter some text first.")
        else:
            # ===== SENTIMENT ANALYSIS =====
            if "Sentiment Analysis" in analysis_options:
                st.subheader("💭 Sentiment Analysis")
                with st.spinner("Analyzing sentiment..."):
                    try:
                        blob = TextBlob(text)
                        sentiment = blob.sentiment
                        
                        # Determine sentiment category
                        if sentiment.polarity > 0.1:
                            sentiment_label = "😊 Positive"
                            sentiment_color = "green"
                        elif sentiment.polarity < -0.1:
                            sentiment_label = "😞 Negative"
                            sentiment_color = "red"
                        else:
                            sentiment_label = "😐 Neutral"
                            sentiment_color = "gray"
                        
                        col_sent1, col_sent2, col_sent3 = st.columns(3)
                        
                        with col_sent1:
                            st.metric("Sentiment", sentiment_label)
                        with col_sent2:
                            st.metric("Polarity", f"{sentiment.polarity:.3f}")
                            st.caption("Range: -1 (negative) to +1 (positive)")
                        with col_sent3:
                            st.metric("Subjectivity", f"{sentiment.subjectivity:.3f}")
                            st.caption("Range: 0 (objective) to 1 (subjective)")
                        
                        # Polarity visualization
                        polarity_percentage = (sentiment.polarity + 1) / 2
                        st.progress(polarity_percentage, 
                                  text=f"Sentiment Scale: {sentiment.polarity:.3f}")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            
            # ===== WORD CLOUD =====
            if "Word Cloud" in analysis_options:
                st.subheader("☁️ Word Cloud")
                with st.spinner("Generating word cloud..."):
                    try:
                        wordcloud = WordCloud(
                            width=800, 
                            height=400, 
                            background_color='white',
                            colormap='viridis',
                            relative_scaling=0.5,
                            min_font_size=10
                        ).generate(text)
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            
            # ===== NAMED ENTITY RECOGNITION =====
            if "Named Entity Recognition (NER)" in analysis_options:
                st.subheader("🏷️ Named Entity Recognition (NER)")
                with st.spinner("Loading NLP model..."):
                    nlp, displacy = load_spacy()
                with st.spinner("Extracting entities..."):
                    try:
                        doc = nlp(text)
                        
                        if doc.ents:
                            # Display entities in a nice format
                            entities_data = []
                            for ent in doc.ents:
                                entities_data.append({
                                    "Text": ent.text,
                                    "Label": ent.label_,
                                    "Description": spacy.explain(ent.label_)
                                })
                            
                            import pandas as pd
                            entities_df = pd.DataFrame(entities_data)
                            
                            st.success(f"✅ Found {len(doc.ents)} entities!")
                            st.dataframe(entities_df, use_container_width=True)
                            
                            # Visualize entities
                            st.markdown("**Entity Visualization:**")
                            html = displacy.render(doc, style="ent", jupyter=False)
                            st.markdown(html, unsafe_allow_html=True)
                        else:
                            st.info("ℹ️ No named entities found in the text.")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            
            # ===== TEXT STATISTICS =====
            if "Text Statistics" in analysis_options:
                st.subheader("📊 Text Statistics")
                try:
                    words = text.split()
                    sentences = text.split('.')
                    characters = len(text)
                    
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    
                    with col_stat1:
                        st.metric("Characters", characters)
                    with col_stat2:
                        st.metric("Words", len(words))
                    with col_stat3:
                        st.metric("Sentences", len([s for s in sentences if s.strip()]))
                    with col_stat4:
                        avg_word_len = sum(len(word) for word in words) / len(words) if words else 0
                        st.metric("Avg Word Length", f"{avg_word_len:.1f}")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("**🧠 AI Data Analysis** | Built with Streamlit, DeepFace, spaCy, and more!")
