import os
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from src.data_loader import load_all_documents, split_documents
from src.embedding import EmbeddingManager
from src.vectoriser import VectorStore


class RAGRetriever:
    """Handles query-based retrieval from the vector store"""
    
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):

        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:

        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")
        
        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        if not isinstance(query_embedding, list):
                query_embedding = query_embedding.tolist()

        # Search in vector store
        try:
            results = self.vector_store.collection.query(
                
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # Process results
            retrieved_docs = []
            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                
                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1 - distance
                    
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })
                
                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found")
            
            return retrieved_docs
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []


if __name__ == '__main__':
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
    print(rag_retriever.retrieve("What is attention is all you need"))
    print("DONE")


