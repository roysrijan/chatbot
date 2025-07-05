from flask import Flask, request, jsonify
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from flask_cors import CORS

# --- Flask App ---
app = Flask(__name__)

# --- Enable CORS --
CORS(app)

# --- Environment Variables ---
# Ensure these are set in your deployment environment
os.environ['LANGSMITH_TRACING'] = os.getenv('LANGSMITH_TRACING', 'false') # Set to true for tracing
os.environ['LANGSMITH_ENDPOINT'] = os.getenv('LANGSMITH_ENDPOINT', 'https://api.smith.langchain.com')
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# --- RAG Setup (Initialize outside the request handler) ---
@app.before_request
def setup_rag():
    global retriever, llm, prompt # Make these global so they can be accessed in the request handler

    # Set embeddings
    if not os.environ.get('OPENAI_API_KEY'):
        print("Warning: OPENAI_API_KEY not set. Embeddings may fail.")
    embd = OpenAIEmbeddings()

    # Docs to index
    urls = [
        "1930.pdf",
        "1946.pdf",
    ]

    # Load document
    # Assuming 'myIRC.pdf' is available in the same directory as app.py
    if not os.path.exists("1930.pdf"):
         print("Warning: '1930.pdf' not found. Document loading will fail.")
         # You might want to handle this error more gracefully in a production app
    else:
        docs = [PyPDFLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]

        # Split
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500, chunk_overlap=0
        )
        doc_splits = text_splitter.split_documents(docs_list)

        # Add to vectorstore
        # Using a persistent vectorstore for production might be better
        # For this example, it's in-memory within the application's lifecycle
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=embd,
        )
        retriever = vectorstore.as_retriever()

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# --- API Endpoint ---
@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({"error": "No query provided"}), 400

    if 'retriever' not in globals() or 'llm' not in globals() or 'prompt' not in globals():
         return jsonify({"error": "RAG components not initialized. Check setup."}), 500

    try:
    q_docs = PyPDFLoader("0107queries.pdf").load()

    # Add to vectorstore
    db = Chroma.from_documents(
        documents=q_docs,
        collection_name="rag-chroma",
        embedding=embd,
    )
    res = db.similarity_search_with_score(query, k=1)
    context_text="\n\n-------\n\n".join([document[0].page_content for document in res])
    start_index = context_text.find("Draft Reply")
    if start_index != -1:
        reply = context_text[start_index + len("Draft Reply"):].strip()
        print(reply)
        return jsonify({
            "answer": reply,
            "sources": {
                "page_content": reply[:150],  # First 150 characters of the reply
                "metadata": {
                    "source": "0107queries.pdf",
                }
            }
        })
    else:
        print("Draft Reply not found in the text.")
        # Chain
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

        result = chain.invoke({"query": query})
        formatted_sources = [
            {"page_content": doc.page_content[:150], "metadata": doc.metadata}
            for i, doc in enumerate(result.get('source_documents'), 1)
        ]
        return jsonify({
            "answer": result.get('result'),
            "sources": formatted_sources
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Health Check Endpoint (Optional but Recommended) ---
@app.route('/health', methods=['GET'])
def health_check():
    # Basic check to see if the application is running
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    # In a production deployment, the hosting platform will typically manage this
    # but this is useful for local testing.
    # Ensure myIRC.pdf is present for local testing
    if not os.path.exists("myIRC.pdf"):
        print("\n" + "="*50)
        print("!!! WARNING: 'myIRC.pdf' not found for local testing. !!!")
        print("!!! The RAG setup will likely fail.                  !!!")
        print("!!! Place 'myIRC.pdf' in the 'rag_api' directory.     !!!")
        print("="*50 + "\n")

    app.run(host='0.0.0.0', port=8080) # Use 0.0.0.0 to be accessible within the container
