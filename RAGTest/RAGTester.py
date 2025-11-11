import json
import re
from difflib import SequenceMatcher
from src.data_loader import load_all_documents, split_documents
from src.embedding import EmbeddingManager
from src.vectoriser import VectorStore
from src.RAGRetriever import RAGRetriever
from src.llm import LLMManager

class RAGTester:
        
    def __init__(self, test_path: str, output_path: str, model: str = "openai", model_name: str = "gpt-4o", use_openai_embed: bool = True):
        self.test_path = test_path
        self.output_path = output_path
        self.model = model
        self.model_name = model_name
        self.test_cases = self._load_test_cases()
        self.embedding_manager = EmbeddingManager(use_openai=use_openai_embed)
        self.vector_store = None
        self.retriever = None
        self.llm = LLMManager(provider=model, model_name=model_name)


    def _load_test_cases(self):
        with open(self.test_path, "r",encoding="utf-8") as f:
            return json.load(f)

    def _compute_coverage(self, answer, expected):
        return SequenceMatcher(None, answer.lower(), expected.lower()).ratio()

    def _check_hallucination(self, answer, context):
        return any(phrase in answer and phrase not in context for phrase in re.findall(r'\b\w+\b', answer))

    def _is_fluent(self, text):
        return bool(re.match(r'^[A-Z].*[.]$', text.strip()))

    def setup_pipeline(self):
        docs = load_all_documents("Data")
        chunks = split_documents(docs)
        texts = [doc.page_content for doc in chunks]

        embeddings = self.embedding_manager.generate_embeddings(texts)
        self.vector_store = VectorStore(collection_name="docs", persist_directory="vector_store")
        self.vector_store.add_documents(chunks, embeddings)
        self.retriever = RAGRetriever(self.vector_store, self.embedding_manager)

    def run(self):
        self.setup_pipeline()
        results = []

        for case in self.test_cases:
            query = case["query"]
            expected = case["expected_answer"]
            source = case["source"]
            difficulty = case["difficulty"]

            retrieved = self.retriever.retrieve(query, top_k=5)
            context = "\n\n".join([doc["content"] for doc in retrieved])
            relevance_scores = [doc["similarity_score"] for doc in retrieved]
            avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0

            response = self.llm.generate_response(query, context)

            coverage = self._compute_coverage(response, expected)
            hallucination = self._check_hallucination(response, context)
            accuracy = 1.0 if coverage > 0.7 else 0.0
            fluency = 1.0 if self._is_fluent(response) else 0.5

            results.append({
                "id": case["id"],
                "query": query,
                "expected_answer": expected,
                "generated_answer": response,
                "source": source,
                "difficulty": difficulty,
                "max_min_relevance": [max(relevance_scores), min(relevance_scores)],
                "avg_relevance": round(avg_relevance, 3),
                "accuracy": accuracy,
                "coverage": round(coverage, 3),
                "hallucination": hallucination,
                "fluency": fluency
            })

        with open(self.output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Evaluation complete. Results saved to {self.output_path}")