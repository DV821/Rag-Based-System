import os
from dotenv import load_dotenv
from src.data_loader import load_all_documents, split_documents
from src.embedding import EmbeddingManager
from src.vectoriser import VectorStore
from src.RAGRetriever import RAGRetriever
from src.llm import LLMManager
from langchain_openai import ChatOpenAI

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# --- Enhanced RAG Pipeline Features ---
def rag_advanced(query, retriever, llm, top_k=5, min_score=0.2, return_context=False):
    """
    RAG pipeline with extra features:
    - Returns answer, sources, confidence score, and optionally full context.
    """
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)
    if not results:
        return {'answer': 'No relevant context found.', 'sources': [], 'confidence': 0.0, 'context': ''}
    
    # Prepare context and sources
    context = "\n\n".join([doc['content'] for doc in results])
    sources = [{
        'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
        'page': doc['metadata'].get('page', 'unknown'),
        'score': doc['similarity_score'],
        'preview': doc['content'][:300] + '...'
    } for doc in results]
    confidence = [doc['similarity_score'] for doc in results]
    
    # Generate answer
    prompt = f"""You are a well qualified AI researcher. Based on the context, answer the questions. \nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"""
    response = llm.invoke([prompt.format(context=context, query=query)])
    
    output = {
        'answer': response.content,
        'sources': sources,
        'confidence': confidence
    }
    if return_context:
        output['context'] = context
    return output


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
    
    llm=ChatOpenAI(model_name="gpt-4o",temperature=0.1,max_tokens=1024)

    print(scores)
    
    result = rag_advanced(query, rag_retriever, llm, top_k=5, min_score=0.1, return_context=True)
    print("Answer:", result['answer'])
    print("Sources:", result['sources'])
    print("Confidence:", result['confidence'])
    print("Context Preview:", result['context'][:300])
    print("DONE")