import os
from flask import Flask, request, jsonify
from langchain.chains import RetrievalQA
from flask_cors import CORS
from rag_service.agent import agent
# from rag_service import llm, retriever, prompt
from langchain_core.messages import convert_to_messages


def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")



# --- Flask App ---
app = Flask(__name__)

# --- Enable CORS --
CORS(app)

# --- API Endpoint ---
@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({"error": "No query provided"}), 400

    if 'retriever' not in globals() or 'llm' not in globals() or 'prompt' not in globals():
         return jsonify({"error": "RAG components not initialized. Check setup."}), 500

    # Chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    try:
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

# --- API Endpoint ---
@app.route('/agent-query', methods=['POST'])
def handle_agent_query():
    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({"error": "No query provided"}), 400

    if 'agent' not in globals():
        return jsonify({"error": "Agent not initialized. Check setup."}), 500

    try:
        for chunk in agent.stream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": query,
                    }
                ]
            },
            {"configurable": {"thread_id": "123"}}, # Added thread_id here
            subgraphs=True,
        ):
            pretty_print_messages(chunk, last_message=True)
        result = chunk[1]["supervisor"]["messages"][-1:][0].content
        return jsonify({"answer": result})
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

    app.run(host='0.0.0.0', port=80) # Use 0.0.0.0 to be accessible within the container
