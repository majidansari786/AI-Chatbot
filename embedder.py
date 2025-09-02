import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredExcelLoader

docx_folder = "./docx"
pdf_folder = "./pdf"
excel_folder = "./excel"
docs = []

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_retriever():
    faiss_path = "faiss_index"
    if os.path.exists(faiss_path):
        print("📂 Loading existing FAISS index...")
        vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    else:
        for filename in os.listdir(pdf_folder):
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(pdf_folder, filename))
                docs.extend(loader.load())
            for filename in os.listdir(docx_folder):
                if filename.endswith(".docx"):
                    loader = Docx2txtLoader(os.path.join(docx_folder, filename))
                    docs.extend(loader.load())
            for filename in os.listdir(excel_folder):
                if filename.endswith(".xlsx") or filename.endswith(".xls"):
                    loader = UnstructuredExcelLoader(os.path.join(excel_folder, filename))
                    docs.extend(loader.load())

            print(f"Loaded {len(docs)} chunks from {pdf_folder} and {docx_folder} and {excel_folder}.")
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            split_docs = splitter.split_documents(docs)
            print(f"🧩 Split into {len(split_docs)} document chunks.")
            print("🔍 Building new FAISS index...")
            vectorstore = FAISS.from_documents(split_docs, embeddings)
            vectorstore.save_local(faiss_path)
            print("💾 FAISS index saved.")
    return vectorstore.as_retriever()
