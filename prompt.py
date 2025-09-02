from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.prompts import ChatPromptTemplate
from embedder import get_retriever
import os
import dotenv

dotenv.load_dotenv()

os.getenv("SERPAPI_API_KEY")
os.getenv("SCENEX_API_KEY")

retriever = get_retriever()

llm = ChatOpenAI(
    model="local-model",              # can be any string
    openai_api_key="lm-studio",       # default LM Studio key
    openai_api_base="http://localhost:3000/v1"
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

custom_prompt = ChatPromptTemplate.from_messages([
    ("user", '''You are Xavier, a helpful AI assistant.
     You will always receive two inputs:
     context → {context}

    question → {question}

    Your rules:
    Use the context to answer the question with as much detail and clarity as possible.
    If the answer is not in the context, then search in the document and if not found then search on internet and still not found then reply with :
    "I don’t know based on the provided context."
    If the question is unrelated to the context, reply with:
    "I’m tuned to only answer questions related to the given context."
    Never invent or assume information not present in the context.
    Your goal: Provide accurate, context-driven, and conversational answers.''')
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
    description="Use this to answer questions about the document."
)

builtin_tools = load_tools(["llm-math","serpapi"], llm=llm)
agent = initialize_agent(
    tools=builtin_tools + [pdf_tool],
    llm=llm,
    agent="conversational-react-description",
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)

def handle_user_query(query: str) -> str:
    """Handles a user query and returns the AI's response."""
    response = agent.invoke({"input": query})
    return response["output"]
