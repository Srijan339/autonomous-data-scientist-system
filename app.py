import streamlit as st
from src.data_processing.data_loader import DataLoader

st.title("Autonomous Data Scientist Platform")

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "json"]
)

if uploaded_file:

    loader = DataLoader()

    df = loader.load_uploaded_file(uploaded_file)

    st.write("Dataset Preview")
    st.dataframe(df.head())