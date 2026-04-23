"""
rag_server.py — Persistent RAG engine HTTP server
===================================================
Runs as a standalone Flask server that Node.js calls via HTTP.
The RAG engine stays in memory so loaded datasets persist.

Usage:
    python rag_server.py [--port 5001]

Endpoints:
    GET  /api/rag/status
    GET  /api/rag/models
    POST /api/rag/upload   (multipart form: file)
    POST /api/rag/chat     (JSON: {question, backend, allowFullContext})
    POST /api/rag/clear
"""

import os
import sys
import json
import argparse
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

sys.path.insert(0, os.path.dirname(__file__))
from rag_engine import RAGEngine

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "rag_data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls", "json", "pdf", "txt"}

rag = RAGEngine()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/api/rag/status", methods=["GET"])
def status():
    return jsonify(rag.get_status())


@app.route("/api/rag/models", methods=["GET"])
def models():
    try:
        import requests as req

        r = req.get("http://localhost:11434/api/tags", timeout=3)
        names = [m["name"] for m in r.json().get("models", [])]
        return jsonify({"models": names, "count": len(names)})
    except Exception as e:
        return jsonify({"models": [], "error": str(e)})


@app.route("/api/rag/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify(
            {"error": "Invalid file type. Allowed: " + ", ".join(ALLOWED_EXTENSIONS)}
        ), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    try:
        result = rag.load_document(filepath, filename)
        return jsonify(result)
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/rag/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question", "").strip()
    backend = data.get("backend", "ollama")
    allow_full_context = bool(data.get("allowFullContext", False))

    if not question:
        return jsonify({"error": "Empty question"}), 400
    if not rag.is_loaded():
        return jsonify({"error": "No dataset loaded. Please upload a file first."}), 400

    try:
        result = rag.ask(
            question,
            backend=backend,
            allow_full_context=allow_full_context,
        )
        if isinstance(result, dict):
            result.setdefault("source", "rag-engine")
            result.setdefault("backend", backend)
            return jsonify(result)
        return jsonify(
            {
                "success": True,
                "source": "rag-engine",
                "answer": result,
                "backend": backend,
            }
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/rag/clear", methods=["POST"])
def clear():
    rag.clear()
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Engine HTTP Server")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()

    print(f"\nRAG Engine Server running on http://localhost:{args.port}\n")
    app.run(debug=False, port=args.port, host="0.0.0.0")
