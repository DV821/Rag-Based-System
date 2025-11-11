import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from src.data_loader import load_all_documents, split_documents
from src.embedding import EmbeddingManager
from src.vectoriser import VectorStore
from src.RAGRetriever import RAGRetriever

load_dotenv()

class LLMManager:
    
    def __init__(self, provider: str = "openai", model_name: str = "gpt-4o", temperature: int = 0.1, max_tokens: int = 1024, api_key: str = None):
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key or self._get_api_key()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = self._load_model()
        
        print(f"Initialized {self.provider.upper()} LLM with model: {self.model_name}")

    def _get_api_key(self):
        keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "groq": os.getenv("GROQ_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY")
        }
        key = keys.get(self.provider)
        if not key:
            raise ValueError(f"{self.provider.upper()} API key is missing. Set the appropriate environment variable.")
        return key

    def _load_model(self):
        if self.provider == "openai":
            return ChatOpenAI(
                model_name=self.model_name,
                temperature=0.1,
                max_tokens=1024,
                openai_api_key=self.api_key
            )
        elif self.provider == "groq":
            return ChatGroq(
                model_name=self.model_name,
                temperature=0.1,
                max_tokens=1024,
                groq_api_key=self.api_key
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate_response(self, query: str, context: str, max_length: int = 500) -> str:
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful AI assistant. Use the following context to answer the question accurately and concisely.

                Context:
                {context}

                Question: {question}

                Answer: Provide a clear and informative answer based on the context above. If the context doesn't contain enough information to answer the question, say so."""
                        )
        
        # Format the prompt
        formatted_prompt = prompt_template.format(context=context, question=query)
        
        try:
            # Generate response
            messages = [HumanMessage(content=formatted_prompt)]
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
        
    def generate_response_simple(self, query: str, context: str) -> str:
        """
        Simple response generation without complex prompting
        
        Args:
            query: User question
            context: Retrieved context
            
        Returns:
            Generated response
        """
        simple_prompt = f"""Based on this context: {context}

Question: {query}

Answer:"""
        
        try:
            messages = [HumanMessage(content=simple_prompt)]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"
    
        
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
    
    query = "What is DeepSeek?"
    retrieved_docs = rag_retriever.retrieve(query, score_threshold=0.5)
    scores = [doc['similarity_score'] for doc in retrieved_docs]
    # print(scores)
    
    # response = llm.generate_response_simple(query, "\n\n".join([doc["content"] for doc in retrieved_docs]))
    response = llm.generate_response(query, "\n\n".join([doc["content"] for doc in retrieved_docs]))

    print(response)
    print(scores)
    print("DONE")