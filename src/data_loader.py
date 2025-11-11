from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported files from the data directory and convert to LangChain document structure.
    Supported: PDF, TXT, CSV, Excel, Word, JSON
    """
    # Use project root data folder
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}\n")
    documents = []
    
    print('-----------------------------------------------------\n')
    # PDF files
    pdf_files = list(data_path.glob('**/*.pdf'))
    print(f"\n[DEBUG] Found {len(pdf_files)} PDF files")
    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading PDF: {pdf_file}")
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} PDF docs from {pdf_file} \n")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load PDF {pdf_file}: {e}")

    # print('-----------------------------------------------------')
    # # Excel files
    # xlsx_files = list(data_path.glob('**/*.xlsx'))
    # print(f"[DEBUG] Found {len(xlsx_files)} Excel files\n")
    # for xlsx_file in xlsx_files:
    #     print(f"[DEBUG] Loading Excel: {xlsx_file}")
    #     try:
    #         loader = PyMuPDFLoader(str(xlsx_file))
    #         loaded = loader.load()
    #         print(f"[DEBUG] Loaded {len(loaded)} Excel docs from {xlsx_file}")
    #         documents.extend(loaded)
    #     except Exception as e:
    #         print(f"[ERROR] Failed to load Excel {xlsx_file}: {e}")

    print('-----------------------------------------------------\n')
    # Word files
    docx_files = list(data_path.glob('**/*.docx'))
    print(f"[DEBUG] Found {len(docx_files)} Word files\n")
    for docx_file in docx_files:
        print(f"[DEBUG] Loading Word: {docx_file}")
        try:
            loader = PyMuPDFLoader(str(docx_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} Word docs from {docx_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load Word {docx_file}: {e}")


    print('-----------------------------------------------------\n')
    
    print(f"[DEBUG] Total loaded documents: {len(documents)}")
    return documents

def split_documents(documents,chunk_size=1000,chunk_overlap=200):
    """Split documents into smaller chunks for better RAG performance"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
    
    # Show example of a chunk
    if split_docs:
        print(f"\nExample chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].metadata}")
    
    return split_docs

# Example usage
if __name__ == "__main__":
    docs = load_all_documents("Data")
    print(f"Loaded {len(docs)} documents.")
    split_docs = split_documents(docs)
    print(split_docs[0])