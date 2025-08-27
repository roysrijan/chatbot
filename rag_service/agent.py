import os
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredExcelLoader
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import convert_to_messages
from langgraph.graph import END, MessagesState, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools.retriever import create_retriever_tool


# --- Environment Variables ---
# Ensure these are set in your deployment environment
os.environ['LANGSMITH_TRACING'] = os.getenv('LANGSMITH_TRACING', 'true') # Set to true for tracing
os.environ['LANGSMITH_ENDPOINT'] = os.getenv('LANGSMITH_ENDPOINT', 'https://api.smith.langchain.com')
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')

web_search = TavilySearch(max_results=3)


research_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[web_search],
    prompt=(
        "You are a research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with research-related tasks, DO NOT do any math or answer tax-related questions\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="research_agent",
)


embd = OpenAIEmbeddings()
urls = [
    "data.xlsx",
]
q_docs = [UnstructuredExcelLoader(url).load() for url in urls]
docs_list = [item for sublist in q_docs for item in sublist]
# Add to vectorstore
db = Chroma.from_documents(
    persist_directory="vector_db",
    documents=docs_list,
    collection_name="rag-chroma",
    embedding=embd,
)
retriever = db.as_retriever()


retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_doc_info",
    "Search and return information about docs.",
)



tax_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[retriever_tool],
    prompt=(
        "You are a tax agent\n\n"
        "INSTRUCTIONS:\n"
        "- Use the following pieces of retrieved context to answer the question\n"
        "- Assist ONLY with tax-related queries, DO NOT do any math or web search\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),

    name="tax_agent",
)

def add(a: float, b: float):
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float):
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float):
    """Divide two numbers."""
    return a / b


math_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[add, multiply, divide],
    prompt=(
        "You are a math agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with math-related tasks\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="math_agent",
)


# Memory for supervisor
supervisor_memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True
)

supervisor = create_supervisor(
    model=init_chat_model("openai:gpt-4.1"),
    agents=[research_agent, math_agent, tax_agent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a tax agent. Assign tax-related tasks to this agent\n"
        "- a research agent. Assign research-related tasks to this agent\n"
        "- a math agent. Assign math-related tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "After answering, ask the user if they have any further questions.\n"
        "Transfer trailing question to previous agent\n"
        "Do not do any work yourself."
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
    memory=supervisor_memory,   # ✅ attach memory
).compile()

def setup_workflow():
    memory = MemorySaver()

    # Define the multi-agent supervisor graph
    workflow = StateGraph(state_schema=MessagesState)

    workflow.add_edge(START, "supervisor")
    workflow.add_node("supervisor", supervisor)

    agent = workflow.compile(checkpointer=memory)
    return agent

agent = setup_workflow()
