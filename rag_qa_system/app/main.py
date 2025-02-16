from flask import Flask, request, jsonify
from retrieval import DocumentRetriever
from generation import AnswerGenerator

app = Flask(__name__)

# Initialize the retriever and generator
retriever = DocumentRetriever(knowledge_base_dir="app/knowledge_base")
generator = AnswerGenerator()

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    query = data.get("query")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    # Retrieve relevant documents
    documents = retriever.retrieve(query, top_k=3)
    context = " ".join(documents)

    # Generate an answer
    answer = generator.generate_answer(context, query)

    return jsonify({"query": query, "answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)