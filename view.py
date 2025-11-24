import streamlit as st
import pandas as pd

BASE_PARQUET_URL = (
    "hf://datasets/bitext/Bitext-retail-banking-llm-chatbot-training-dataset/"
    "bitext-retail-banking-llm-chatbot-training-dataset.parquet"
)

df = pd.read_parquet(BASE_PARQUET_URL)

st.title("Bitext Banking Dataset — Categories & Intents")

categories = sorted(df["category"].dropna().unique())
st.subheader(f"All Categories ({len(categories)})")
st.dataframe(pd.DataFrame({"category": categories}), use_container_width=True)

st.subheader("Intents per Category")
grouped = (
    df[["category", "intent"]]
    .dropna()
    .drop_duplicates()
    .sort_values(["category", "intent"])
)

for cat, sub in grouped.groupby("category"):
    st.markdown(f"### {cat}")
    st.write(sorted(sub["intent"].unique()))
