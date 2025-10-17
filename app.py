import streamlit as st
import pandas as pd
import plotly.express as px
from gtts import gTTS
import os
import whisper

st.set_page_config(layout="wide")  # Full-width layout
st.title("Placement Data Analytics Dashboard")

# Read CSV
df = pd.read_csv("data.csv")

# --- Summary Cards ---
total_students = len(df)
total_branches = df['Branch'].nunique()
total_recruiters = df['Name of the Employer'].nunique()

col1, col2, col3 = st.columns(3)
col1.metric("Total Students", total_students)
col2.metric("Unique Branches", total_branches)
col3.metric("Total Recruiters", total_recruiters)

st.markdown("---")  # Horizontal line

# --- Side-by-side graphs ---
col1, col2 = st.columns(2)

# 1️⃣ Bar chart: Year-wise student count
year_counts = df['Year'].value_counts().sort_index()
fig_bar = px.bar(
    x=year_counts.index,
    y=year_counts.values,
    color = year_counts.index,
    labels={'x': 'Year', 'y': 'Number of Students'},
    title="Year-wise Placement Count"
)
col1.plotly_chart(fig_bar, use_container_width=True)

# 2️⃣ Pie chart: Branch-wise distribution
year_counts = df['Year'].value_counts()
fig_pie = px.pie(
    names=year_counts.index,
    values=year_counts.values,
    title="year-wise Distribution"
)
col2.plotly_chart(fig_pie, use_container_width=True)


st.markdown("---")  # separator
# --- Treemap: Branch-wise placement ---

branch_counts = df['Branch'].value_counts().reset_index()
branch_counts.columns = ['Branch', 'Count']

fig_treemap = px.treemap(
    branch_counts,
    path=['Branch'],  # hierarchy levels, here only Branch
    values='Count',
    title="Branch-wise Placement Treemap"
)

st.plotly_chart(fig_treemap, use_container_width=True)

st.markdown("---")

years = sorted(df['Year'].unique())

# Option to choose all years or selected year
year_option = st.selectbox("Select Year for Top Recruiters", ["All"] + years, index=len(years))

if year_option == "All":
    df_top = df
else:
    df_top = df[df['Year'] == year_option]

# Calculate top 10 recruiters
top_recruiters = df_top['Name of the Employer'].value_counts().head(10).reset_index()
top_recruiters.columns = ['Employer', 'Count']

# Plotly bar chart
fig_top = px.bar(
    top_recruiters,
    x='Count',
    y='Employer',
    color= 'Employer',
    orientation='h',
    title=f"Top 10 Recruiters ({year_option})",
    text='Count'
)
fig_top.update_layout(yaxis={'categoryorder':'total ascending'}, height=500)
st.plotly_chart(fig_top, use_container_width=True)


st.markdown("---")



# --- Sunburst Chart: Year -> Branch -> Employer ---

fig_sunburst = px.sunburst(
    df,
    path=['Year', 'Branch', 'Name of the Employer'],
    values=None,
    color='Branch',
    title="Placements: Year → Branch → Employer",
    color_discrete_sequence=px.colors.qualitative.Pastel
)

# Update layout for bigger chart
fig_sunburst.update_layout(
    margin=dict(t=50, l=25, r=25, b=25),  # adjust margins
    height=800,  # make taller
    width=1200   # make wider (optional)
)

st.plotly_chart(fig_sunburst, use_container_width=True)

st.markdown("---")



st.subheader("Top 10 Recruiters")




# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")
years = sorted(df['Year'].unique())
selected_year = st.sidebar.selectbox("Select Year", years, index=len(years)-1)

branches = df['Branch'].unique()
selected_branch = st.sidebar.multiselect("Select Branch", branches, default=branches)

employers = df['Name of the Employer'].unique()
selected_employer = st.sidebar.multiselect("Select Employer", employers, default=employers)

# Filtered Data
filtered_df = df[
    (df['Year'] == selected_year) &
    (df['Branch'].isin(selected_branch)) &
    (df['Name of the Employer'].isin(selected_employer))
]




# -----------------------------
# Summary Cards
# -----------------------------
total_students = len(filtered_df)
total_branches = filtered_df['Branch'].nunique()
total_recruiters = filtered_df['Name of the Employer'].nunique()

col1, col2, col3 = st.columns(3)
col1.metric("Total Students", total_students)
col2.metric("Unique Branches", total_branches)
col3.metric("Recruiters", total_recruiters)

st.markdown("---")




# -----------------------------
# Side-by-side: Branch and Employer Distributions
# -----------------------------
st.subheader("Branch and Employer Distribution")
col1, col2 = st.columns(2)

# Branch Pie
branch_counts = filtered_df['Branch'].value_counts().reset_index()
branch_counts.columns = ['Branch', 'Count']
fig_branch = px.pie(
    branch_counts, names='Branch', values='Count',
    title="Branch-wise Distribution",
    color_discrete_sequence=px.colors.qualitative.Pastel
)
col1.plotly_chart(fig_branch, use_container_width=True)

# Employer Bar
employer_counts = filtered_df['Name of the Employer'].value_counts().reset_index()
employer_counts.columns = ['Employer', 'Count']
fig_employer = px.bar(
    employer_counts, x='Employer', y='Count',
    title="Top Recruiters",
    text='Count'
)
col2.plotly_chart(fig_employer, use_container_width=True)





# --- Full data table ---
st.subheader("Full Placement Data")
st.dataframe(df, use_container_width=True)

st.markdown("---")

# -----------------------------
# Text-to-Speech Feature
# -----------------------------
st.header("🗣️ Text to Speech")
st.write("Convert any text to speech and listen to it instantly!")

text_input = st.text_area("Enter text to convert to speech:", height=150, placeholder="Type or paste your text here...")

if st.button("🎵 Convert to Audio", type="primary"):
    if text_input.strip():
        with st.spinner("Converting text to speech..."):
            try:
                tts = gTTS(text_input, lang='en')
                audio_path = "output.mp3"
                tts.save(audio_path)
                
                with open(audio_path, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/mp3')
                
                st.success("✅ Conversion complete! Play the audio above.")
                
                # Clean up the file after reading
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except Exception as e:
                st.error(f"❌ Error during conversion: {str(e)}")
    else:
        st.warning("⚠️ Please enter some text to convert.")

st.markdown("---")

# -----------------------------
# Speech-to-Text Feature
# -----------------------------
st.header("🎤 Speech to Text")
st.write("Upload an audio file and convert it to text using Whisper AI!")

@st.cache_resource(show_spinner=False)
def load_whisper_model(model_name="base"):
    """Load and cache the Whisper model"""
    return whisper.load_model(model_name)

# Load model with status indicator
with st.spinner("Loading Whisper model..."):
    try:
        model = load_whisper_model("base")
        st.success("✅ Whisper model loaded and ready!")
    except Exception as e:
        st.error(f"❌ Error loading Whisper model: {str(e)}")
        model = None

uploaded_audio = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "flac", "ogg"], help="Supported formats: WAV, MP3, M4A, FLAC, OGG")

if uploaded_audio:
    # Save uploaded audio temporarily
    temp_file_path = "temp_audio." + uploaded_audio.name.split(".")[-1]
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_audio.read())
    
    st.audio(temp_file_path)
    st.info(f"📁 File uploaded: {uploaded_audio.name}")
    
    if st.button("🎯 Transcribe Audio", type="primary"):
        if model:
            with st.spinner("Transcribing audio with Whisper AI..."):
                try:
                    result = model.transcribe(temp_file_path)
                    
                    st.success("✅ Transcription complete!")
                    st.subheader("📝 Transcribed Text")
                    st.write(result["text"])
                    
                    # Optional: Show detected language
                    if "language" in result:
                        st.info(f"🌐 Detected language: {result['language']}")
                    
                    # Clean up temp file
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                        
                except Exception as e:
                    st.error(f"❌ Error during transcription: {str(e)}")
        else:
            st.error("❌ Whisper model not loaded. Please refresh the page.")



