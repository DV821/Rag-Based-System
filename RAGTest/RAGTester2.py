import json
import re
import string
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.data_loader import load_all_documents, split_documents
from src.embedding import EmbeddingManager
from src.vectoriser import VectorStore
from src.RAGRetriever import RAGRetriever
from src.llm import LLMManager
from src.AdvancedRAGPipeline import AdvancedRAGPipeline

class RAGTester:
    def __init__(
        self,
        test_path: str,
        output_path: str,
        model: str = "openai",
        model_name: str = "gpt-4o",
        use_openai_embed: bool = True,
        use_advanced: bool = False,
        answer_llm=None
    ):
        self.test_path = test_path
        self.output_path = output_path
        self.model = model
        self.model_name = model_name
        self.use_advanced = use_advanced
        self.test_cases = self._load_test_cases()
        self.embedding_manager = EmbeddingManager(use_openai=use_openai_embed)
        self.vector_store = None
        self.retriever = None
        self.llm = LLMManager(provider=model, model_name=model_name)
        self.pipeline = None
        self.answer_llm = answer_llm

    def _load_test_cases(self):
        with open(self.test_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _compute_coverage(self, answer, expected):
        return SequenceMatcher(None, answer.lower(), expected.lower()).ratio()

    def _extract_keywords(self, text: str) -> set:
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = {
            w.strip(string.punctuation)
            for w in words
            if w not in ENGLISH_STOP_WORDS and len(w) > 2
        }
        return keywords

    def _hallucination_score(self, answer: str, context: str) -> float:
        answer_keywords = self._extract_keywords(answer)
        context_keywords = self._extract_keywords(context)

        if not answer_keywords:
            return 1.0  # fully hallucinated if no meaningful answer

        unsupported = answer_keywords - context_keywords
        score = len(unsupported) / len(answer_keywords)
        return round(score, 3)

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

        if self.use_advanced:
            self.pipeline = AdvancedRAGPipeline(self.retriever, self.answer_llm)

    def run(self):
        self.setup_pipeline()
        results = []

        for case in self.test_cases:
            query = case["query"]
            expected = case["expected_answer"]
            source = case["source"]
            difficulty = case["difficulty"]

            if self.use_advanced:
                result = self.pipeline.query(
                    question=query,
                    top_k=5,
                    min_score=0.2,
                    stream=False,
                    summarize=True
                )
                answer = result["summary"]
                context = "\n\n".join([doc["preview"] for doc in result["sources"]])
                relevance_scores = [doc["score"] for doc in result["sources"]]
            else:
                retrieved = self.retriever.retrieve(query, top_k=5)
                context = "\n\n".join([doc["content"] for doc in retrieved])
                relevance_scores = [doc["similarity_score"] for doc in retrieved]
                response = self.llm.generate_response(query, context)
                answer = response

            avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
            coverage = self._compute_coverage(answer, expected)
            hallucination_score = self._hallucination_score(answer, context)
            accuracy = 1.0 if coverage > 0.7 else 0.0
            fluency = 1.0 if self._is_fluent(answer) else 0.5

            results.append({
                "id": case["id"],
                "query": query,
                "expected_answer": expected,
                "generated_answer": answer,
                "source": source,
                "difficulty": difficulty,
                "max_min_relevance": [max(relevance_scores), min(relevance_scores)],
                "avg_relevance": round(avg_relevance, 3),
                "accuracy": accuracy,
                "coverage": round(coverage, 3),
                "hallucination_score": hallucination_score,
                "fluency": fluency
            })

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"✅ Evaluation complete. Results saved to {self.output_path}")