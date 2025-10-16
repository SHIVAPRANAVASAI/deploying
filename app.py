import streamlit as st
import pandas as pd
import plotly.express as px

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
