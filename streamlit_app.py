import os
from dotenv import load_dotenv
import streamlit as st
from src.data_loader import load_all_documents, split_documents
from src.embedding import EmbeddingManager
from src.vectoriser import VectorStore
from src.RAGRetriever import RAGRetriever
from src.llm import LLMManager
from src.AdvancedRAGPipeline import AdvancedRAGPipeline
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="RAG Query App", layout="wide")
st.title("🧠 RAG Query Interface")

# Sidebar config
st.sidebar.header("⚙️ Configuration")
model = st.sidebar.selectbox("Model Provider", ["openai", "groq"])

model_name = ['gpt-4o' if model == 'openai' else "llama-3.1-8b-instant"][0]
use_advanced = st.sidebar.checkbox("Use AdvancedRAGPipeline", value=True)

@st.cache_resource
def initialize_pipeline(model, model_name, use_openai_embed = True):
    embedding_manager = EmbeddingManager(use_openai=use_openai_embed)
    vector_store = VectorStore(collection_name="docs", persist_directory="vector_store")

    # Only embed and add documents if store is empty
    # if not vector_store.has_documents():
    docs = load_all_documents("Data")
    chunks = split_documents(docs)
    texts = [doc.page_content for doc in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)
    vector_store.add_documents(chunks, embeddings)

    retriever = RAGRetriever(vector_store, embedding_manager)
    llm = LLMManager(provider=model, model_name=model_name)
    return retriever, llm

# Query input
query = st.text_area("🔍 Enter your query", height=100)

if st.button("Run RAG"):
    retriever, llm = initialize_pipeline(model, model_name)

    if use_advanced:
        if model == 'openai':
            answer_llm = ChatOpenAI(model_name=model_name, temperature=0.1, max_tokens=1024)
        else:
            answer_llm = ChatGroq(model_name=model_name, temperature=0.1, max_tokens=1024)
        pipeline = AdvancedRAGPipeline(retriever, answer_llm)
        result = pipeline.query(query, top_k=5, min_score=0.2, stream=False, summarize=True)

        st.subheader("📘 Generated Answer")
        st.write(result["answer"])

        # st.subheader("📎 Retrieved Sources")
        
        # seen = set()
        # unique_sources = []
        # for src, page in result["sources"]:
        #     print(src)
        #     key = (src["source"], src["page"])
        #     if key not in seen:
        #         seen.add(key)
        #         unique_sources.append(src)

        # for src, page in unique_sources:
        #     filename = os.path.basename(src["source"])
        #     st.markdown(f"- **{filename}** (page {page}))")
            
        if result["summary"]:
            st.subheader("📝 Summary")
            st.write(result["summary"])
    else:
        retrieved_docs = retriever.retrieve(query, top_k=5)
        context = "\n\n".join([doc["content"] for doc in retrieved_docs])
        response = llm.generate_response(query, context)

        st.subheader("📘 Generated Answer")
        st.write(response)
