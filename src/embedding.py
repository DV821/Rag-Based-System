import os
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
from src.data_loader import load_all_documents, split_documents
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")



class EmbeddingManager:
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_openai: bool = False):
        """
        Initialize the embedding manager.

        Args:
            model_name (str): Name of the embedding model.
            use_openai (bool): Whether to use OpenAIEmbeddings instead of HuggingFace.
        """
        self.model_name = model_name
        self.use_openai = use_openai
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the embedding model."""
        try:
            if self.use_openai:
                print("Loading OpenAIEmbeddings...")
                self.model = OpenAIEmbeddings()
                print("OpenAIEmbeddings loaded successfully.")
            else:
                print(f"Loading HuggingFace model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                print("HuggingFaceEmbeddings loaded successfully.")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> Union[np.ndarray, List[List[float]]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts (List[str]): List of input strings.

        Returns:
            np.ndarray or List[List[float]]: Embedding vectors.
        """
        if not self.model:
            raise ValueError("Embedding model not loaded.")

        print(f"Generating embeddings for {len(texts)} texts...")

        if self.use_openai:
            embeddings = self.model.embed_documents(texts)
        else:
            embeddings = self.model.encode(texts)

        print(f"Generated embeddings for {len(embeddings)} texts.")
        return embeddings
    
## initialize the embedding manager
if __name__ == '__main__':
    docs = load_all_documents("Data")
    print(f"Loaded {len(docs)} documents.")
    chunks = split_documents(docs)
    texts=[doc.page_content for doc in chunks]

    embedding_manager=EmbeddingManager(use_openai=True)
    embeddings=embedding_manager.generate_embeddings(texts)
    print(embeddings[0])
    print("embeddings done")
