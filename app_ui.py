import streamlit as st
import requests
import os

st.set_page_config(page_title="RAG Query App", layout="wide")
st.title("🧠 RAG Query Interface")

# Sidebar config
st.sidebar.header("⚙️ Configuration")
model_name = st.sidebar.selectbox("Model Name", ["gpt-4o", "llama3-8b", "mixtral-8x7b"])
use_advanced = st.sidebar.checkbox("Use AdvancedRAGPipeline", value=True)

# Query input
query = st.text_area("🔍 Enter your query", height=100)

if st.button("Run RAG"):
    response = requests.post("http://localhost:5000/query", json={
        "query": query,
        "use_advanced": use_advanced,
        "model_name": model_name
    })

    if response.status_code == 200:
        result = response.json()
        st.subheader("📘 Generated Answer")
        st.write(result["answer"])

        if result["summary"]:
            st.subheader("📝 Summary")
            st.write(result["summary"])
        
        if use_advanced:
            st.subheader("📎 Retrieved Sources")

            unique_sources = set([source[0] for source in result["sources"]])            

            for source in unique_sources:
                filename = os.path.basename(source)
                st.markdown(f"- **{filename}**")

            
    else:
        st.error("Failed to get response from backend.")