from flask import Flask, request, jsonify
from src.data_loader import load_all_documents, split_documents
from src.embedding import EmbeddingManager
from src.vectoriser import VectorStore
from src.RAGRetriever import RAGRetriever
from src.llm import LLMManager
from src.AdvancedRAGPipeline import AdvancedRAGPipeline
from langchain_openai import ChatOpenAI
import os

app = Flask(__name__)

# Initialize pipeline once
embedding_manager = EmbeddingManager(use_openai=True)
vector_store = VectorStore(collection_name="docs", persist_directory="vector_store")

if not os.path.exists("vector_store") or not os.listdir("vector_store"):
    docs = load_all_documents("Data")
    chunks = split_documents(docs)
    texts = [doc.page_content for doc in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)
    vector_store.add_documents(chunks, embeddings)

retriever = RAGRetriever(vector_store, embedding_manager)

@app.route("/query", methods=["POST"])
def query_rag():
    data = request.json
    query = data.get("query")
    use_advanced = data.get("use_advanced", False)
    model_name = data.get("model_name", "gpt-4o")

    llm = LLMManager(provider="openai", model_name=model_name)

    if use_advanced:
        answer_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.1, max_tokens=1024)
        pipeline = AdvancedRAGPipeline(retriever, answer_llm)
        result = pipeline.query(query, top_k=5, min_score=0.2, stream=False, summarize=True)
        sources = list({(src["source"], src["page"]) for src in result["sources"]})
        return jsonify({
            "answer": result["answer"],
            "summary": result.get("summary", ""),
            "sources": sources
        })
    else:
        retrieved_docs = retriever.retrieve(query, top_k=5)
        context = "\n\n".join([doc["content"] for doc in retrieved_docs])
        answer = llm.generate_response(query, context)
        # sources = list({(doc["source"], doc["page"]) for doc in retrieved_docs})
        return jsonify({
            "answer": answer,
            "summary": "",
            # "sources": sources
        })

if __name__ == "__main__":
    app.run(debug=True)