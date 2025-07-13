import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain import hub

# Use DebugOpenAIEmbeddings if needed
class DebugOpenAIEmbeddings(OpenAIEmbeddings):
    def embed_documents(self, texts, chunk_size=1000):
        print(f"🔍 Embedding {len(texts)} chunks")
        return super().embed_documents(texts, chunk_size=chunk_size)


# --- Environment Variables ---
# Ensure these are set in your deployment environment
os.environ['LANGSMITH_TRACING'] = os.getenv('LANGSMITH_TRACING', 'true') # Set to true for tracing
os.environ['LANGSMITH_ENDPOINT'] = os.getenv('LANGSMITH_ENDPOINT', 'https://api.smith.langchain.com')
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


def setup_rag():
    # Setup embeddings
    if not os.environ.get('OPENAI_API_KEY'):
        print("Warning: OPENAI_API_KEY not set.")

    embd = DebugOpenAIEmbeddings()

    if not os.path.exists("myIRC.pdf"):
        print("Warning: 'myIRC.pdf' not found.")
        return None, None, None

    loader = PyPDFLoader("myIRC.pdf")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        persist_directory="vector_db",
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embd,
    )
    retriever = vectorstore.as_retriever()

    # Load prompt and LLM
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    return retriever, llm, prompt


retriever, llm, prompt = setup_rag()
