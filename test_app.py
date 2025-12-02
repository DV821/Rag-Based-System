import os
from dotenv import load_dotenv
from RAGTest.RAGTester2 import RAGTester
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def main():
    # Configuration
    test_path = "./RAGTest/test_cases2.json"
    openai_output_path1 = "./RAGTest/openai_evaluation_results1.json"
    openai_output_path2 = "./RAGTest/openai_evaluation_results_adv1.json"
    groq_output_path1 = "./RAGTest/groq_evaluation_results1.json"
    groq_output_path2 = "./RAGTest/groq_evaluation_results_adv1.json"

    # Testing with Open AI Model
    openai_tester = RAGTester(
        test_path=test_path,
        output_path=openai_output_path1,
        model="openai", 
        model_name="gpt-4o", 
        use_openai_embed=True,
        use_advanced=False,
        answer_llm = ChatOpenAI(model_name="gpt-4o",temperature=0.1,max_tokens=1024)
    )
    openai_tester_adv = RAGTester(
        test_path=test_path,
        output_path=openai_output_path2,
        model="openai", 
        model_name="gpt-4o", 
        use_openai_embed=True,
        use_advanced=True,
        answer_llm = ChatOpenAI(model_name="gpt-4o",temperature=0.1,max_tokens=1024)
    )
    
    # Testing with Groq Model
    groq_tester = RAGTester(
        test_path=test_path,
        output_path=groq_output_path1,
        model="groq", 
        model_name="llama-3.1-8b-instant", 
        use_openai_embed=True,
        use_advanced=False, 
        answer_llm = ChatGroq(model_name="llama-3.1-8b-instant",temperature=0.1,max_tokens=1024)
    )
    groq_tester_adv = RAGTester(
        test_path=test_path,
        output_path=groq_output_path2,
        model="groq", 
        model_name="llama-3.1-8b-instant", 
        use_openai_embed=True,
        use_advanced=False, 
        answer_llm = ChatGroq(model_name="llama-3.1-8b-instant",temperature=0.1,max_tokens=1024)
    )
    
    #Running all the testers
    openai_tester.run()
    print("--------------------Open AI Test Finished--------------------")
    openai_tester_adv.run()
    print("--------------------Open AI Adv Test Finished--------------------")
    groq_tester.run()
    print("--------------------Groq Test Finished--------------------")
    groq_tester_adv.run()
    print("--------------------Groq Adv Test Finished--------------------")

if __name__ == "__main__":
    main()