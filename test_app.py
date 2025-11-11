import os
from dotenv import load_dotenv
from RAGTest.RAGTester import RAGTester
from RAGTest.RAGTester2 import RAGTester
from langchain_openai import ChatOpenAI


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def main():
    # Configuration
    test_path = "./RAGTest/test_cases2.json"
    output_path = "./RAGTest/evaluation_results3.json"

    # Choose provider and model
    model = "openai"  # or "groq"
    model_name = "gpt-4o"  # or "llama3-8b", "mixtral-8x7b"

    # Run the test
    # tester = RAGTester(
    #     test_path=test_path,
    #     output_path=output_path,
    #     model=model,
    #     model_name=model_name,
    #     use_openai_embed = True
    # )
    # tester.run()
    
    tester = RAGTester(
        test_path=test_path,
        output_path=output_path,
        model="groq",  # or "openai"
        model_name="llama3-8b",  # or "gpt-4o"
        use_openai_embed=True,
        use_advanced=True,
        answer_llm = ChatOpenAI(model_name="gpt-4o",temperature=0.1,max_tokens=1024)
    )
    tester.run()

if __name__ == "__main__":
    main()