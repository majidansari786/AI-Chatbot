from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, UnstructuredExcelLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain_community.agent_toolkits.load_tools import load_tools
import tiktoken
from langchain.prompts import ChatPromptTemplate
from langchain_text_splitters import TokenTextSplitter
import glob
import os

os.environ["SERPAPI_API_KEY"] = "your_serpapi_key"

def load_documents_from_directory(directory_path):
    """Load documents from a directory supporting multiple file formats"""
    documents = []
    
    # Support file patterns for different formats
    file_patterns = {
        '*.pdf': PyPDFLoader,
        '*.txt': TextLoader,
        '*.csv': CSVLoader,
        '*.docx': UnstructuredWordDocumentLoader,
        '*.xlsx': UnstructuredExcelLoader,
        '*.xls': UnstructuredExcelLoader
    }
    
    for pattern, loader_class in file_patterns.items():
        for file_path in glob.glob(os.path.join(directory_path, pattern)):
            try:
                if pattern == '*.txt':
                    loader = loader_class(file_path, encoding='utf-8')
                else:
                    loader = loader_class(file_path)
                documents.extend(loader.load())
                print(f"Loaded {pattern.replace('*.', '').upper()} file: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
    
    return documents

# Load documents from both the existing PDF directory and the new data directory
all_documents = []

# Load existing PDF files
if os.path.exists("./pdf"):
    all_documents.extend(load_documents_from_directory("./pdf"))

# Load new multi-format files
if os.path.exists("./data/samples"):
    all_documents.extend(load_documents_from_directory("./data/samples"))

if not all_documents:
    print("No documents found. Please add documents to ./pdf/ or ./data/samples/ directory")
    exit(1)

text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separator="\n"
)
doc = text_splitter.split_documents(all_documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectrstore = FAISS.from_documents(doc, embeddings)

retriever = vectrstore.as_retriever()


llm = ChatOpenAI(
    model="local-model",              # can be any string
    openai_api_key="lm-studio",       # default LM Studio key
    openai_api_base="http://localhost:3000/v1"
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
custom_prompt = ChatPromptTemplate.from_messages([
    ("user", "Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}")
])
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt}
)
document_tool = Tool(
    name="Document Knowledge Base",
    func=lambda q: qa_chain.run(q),
    description="Use this to answer questions about uploaded documents (PDF, TXT, DOCX, Excel, CSV files)."
)

system_message = """
You are a helpful AI assistant.
You have access to the following tools: Calculator, Document Knowledge Base.
You **must** use the 'Calculator' tool for any mathematical calculation, no matter how simple it seems.
Do not perform calculations yourself. Always delegate math to the 'Calculator' tool. And when user asking from Document Knowledge Base, you must use the 'Document Knowledge Base' tool.
The Document Knowledge Base contains information from various document formats including PDF, TXT, DOCX, Excel, and CSV files.
"""

builtin_tools = load_tools(["llm-math","serpapi"], llm=llm)
agent = initialize_agent(
    tools=builtin_tools + [document_tool],
    llm=llm,
    agent="conversational-react-description",
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)



while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    response = agent.invoke({"input": query})
    print("Agent:", response["output"])
