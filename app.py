from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
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

import os

os.environ["SERPAPI_API_KEY"] = "your_serpapi_key"

loader = PyPDFLoader("./pdf/sample.pdf")
documents =loader.load()
text_splitter = TokenTextSplitter(
    encoding_name="cl100k_base",
    chunk_size=500,
    chunk_overlap=50
)
doc = text_splitter.split_documents(documents)

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
pdf_tool = Tool(
    name="PDF Knowledge Base",
    func=lambda q: qa_chain.run(q),
    description="Use this to answer questions about the PDF document."
)

system_message = """
You are a helpful AI assistant.
You have access to the following tools: Calculator, PDF Knowledge Base.
You **must** use the 'Calculator' tool for any mathematical calculation, no matter how simple it seems.
Do not perform calculations yourself. Always delegate math to the 'Calculator' tool. And when user asking from PDF Knowledge Base, you must use the 'PDF Knowledge Base' tool.
"""

builtin_tools = load_tools(["llm-math","serpapi"], llm=llm)
agent = initialize_agent(
    tools=builtin_tools + [pdf_tool],
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
