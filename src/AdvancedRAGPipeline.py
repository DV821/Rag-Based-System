import os
from dotenv import load_dotenv
from src.data_loader import load_all_documents, split_documents
from src.embedding import EmbeddingManager
from src.vectoriser import VectorStore
from src.RAGRetriever import RAGRetriever
from src.llm import LLMManager
from langchain_openai import ChatOpenAI
# --- Advanced RAG Pipeline: Streaming, Citations, History, Summarization ---
from typing import Dict, Any
import time

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class AdvancedRAGPipeline:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.history = []  # Store query history

    def query(self, question: str, top_k: int = 5, min_score: float = 0.2, stream: bool = False, summarize: bool = False) -> Dict[str, Any]:
        # Retrieve relevant documents
        results = self.retriever.retrieve(question, top_k=top_k, score_threshold=min_score)
        if not results:
            answer = "No relevant context found."
            sources = []
            context = ""
        else:
            context = "\n\n".join([doc['content'] for doc in results])
            sources = [{
                'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
                'page': doc['metadata'].get('page', 'unknown'),
                'score': doc['similarity_score'],
                'preview': doc['content'][:120] + '...'
            } for doc in results]
            # Streaming answer simulation
            prompt = f"""Use the following context to answer the question concisely.\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"""
            if stream:
                print("Streaming answer:")
                for i in range(0, len(prompt), 80):
                    print(prompt[i:i+80], end='', flush=True)
                    time.sleep(0.05)
                print()
            response = self.llm.invoke([prompt.format(context=context, question=question)])
            answer = response.content

        # Add citations to answer
        citations = [f"[{i+1}] {src['source']} (page {src['page']})" for i, src in enumerate(sources)]
        answer_with_citations = answer + "\n\nCitations:\n" + "\n".join(citations) if citations else answer

        # Optionally summarize answer
        summary = None
        if summarize and answer:
            summary_prompt = f"Summarize the following answer in 1 sentence:\n{answer}"
            summary_resp = self.llm.invoke([summary_prompt])
            summary = summary_resp.content

        # Store query history
        self.history.append({
            'question': question,
            'answer': answer,
            'answer_with_citation': answer_with_citations,
            'sources': sources,
            'summary': summary
        })

        return {
            'question': question,
            'answer': answer,
            'answer_with_citation': answer_with_citations,
            'sources': sources,
            'summary': summary,
            'history': self.history
        }
    
if __name__ == '__main__':
    try:
        # llm = LLMManager(provider="groq", model_name="llama-3.1-8b-instant")
        llm = LLMManager(provider="openai", model_name="gpt-4o")

        print("LLM initialized successfully!")
    except ValueError as e:
        print(f"Warning: {e}")
        print(f"Initialization error: {e}")

    docs = load_all_documents("Data")
    print(f"Loaded {len(docs)} documents.")
    chunks = split_documents(docs)
    texts=[doc.page_content for doc in chunks]

    embedding_manager=EmbeddingManager(use_openai=True)
    embeddings=embedding_manager.generate_embeddings(texts)
    print("embeddings done")
    
    vectorstore=VectorStore(collection_name='docs', persist_directory='vector_store')
    print("vectorstore created")
    vectorstore.add_documents(chunks,embeddings)
    
    
    rag_retriever=RAGRetriever(vectorstore,embedding_manager)
    
    query = "What is multi-head attention??"
    retrieved_docs = rag_retriever.retrieve(query, score_threshold=0.5)
    scores = [doc['similarity_score'] for doc in retrieved_docs]
    
    llm_openAI=ChatOpenAI(model_name="gpt-4o",temperature=0.1,max_tokens=1024)

    print(scores)
# Example usage:
    adv_rag = AdvancedRAGPipeline(rag_retriever, llm_openAI)
    result = adv_rag.query(query, top_k=3, min_score=0.1, stream=True, summarize=True)
    print("\nFinal Answer:", result['answer'])
    print("Summary:", result['summary'])
    print("History:", result['history'][-1])
    print("---------------------------------------------------------------------------------------")
    print(result)