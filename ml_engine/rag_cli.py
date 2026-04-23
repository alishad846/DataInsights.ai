"""
rag_cli.py — CLI bridge for Node.js to call the RAG engine
============================================================
Usage:
  python rag_cli.py --action load --file <path> --filename <name>
  python rag_cli.py --action ask --question "What is total sales?" [--backend ollama] [--allow-full-context]
  python rag_cli.py --action status
  python rag_cli.py --action clear
  python rag_cli.py --action models

Output: JSON to stdout for Node.js to parse.
"""

import argparse
import json
import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(__file__))

from rag_engine import RAGEngine

# Singleton engine instance (persists across calls when used as module)
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine


def action_load(filepath, filename):
    engine = get_engine()
    try:
        result = engine.load_document(filepath, filename)
        print(json.dumps(result, ensure_ascii=True, default=str))
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=True))
        sys.exit(1)


def action_ask(question, backend="ollama", allow_full_context=False):
    engine = get_engine()
    if not engine.is_loaded():
        print(json.dumps({"error": "No dataset loaded. Please upload a file first."}, ensure_ascii=True))
        sys.exit(1)
    try:
        result = engine.ask(question, backend=backend, allow_full_context=allow_full_context)
        if isinstance(result, dict):
            result.setdefault("backend", backend)
            result.setdefault("source", "rag-engine")
            print(json.dumps(result, ensure_ascii=True, default=str))
        else:
            print(json.dumps({
                "success": True,
                "answer": result,
                "backend": backend,
                "source": "rag-engine",
            }, ensure_ascii=True))
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=True))
        sys.exit(1)


def action_status():
    engine = get_engine()
    status = engine.get_status()
    print(json.dumps(status, ensure_ascii=True, default=str))


def action_clear():
    engine = get_engine()
    engine.clear()
    print(json.dumps({"status": "cleared"}))


def action_models():
    try:
        import requests

        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        names = [m["name"] for m in r.json().get("models", [])]
        print(json.dumps({"models": names, "count": len(names)}, ensure_ascii=True))
    except Exception as e:
        print(json.dumps({"models": [], "error": str(e)}, ensure_ascii=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Engine CLI Bridge")
    parser.add_argument(
        "--action", required=True, choices=["load", "ask", "status", "clear", "models"]
    )
    parser.add_argument("--file", default=None)
    parser.add_argument("--filename", default=None)
    parser.add_argument("--question", default=None)
    parser.add_argument("--backend", default="ollama")
    parser.add_argument("--allow-full-context", action="store_true")

    args = parser.parse_args()

    if args.action == "load":
        if not args.file or not args.filename:
            print(
                json.dumps({"error": "--file and --filename required for load action"}, ensure_ascii=True)
            )
            sys.exit(1)
        action_load(args.file, args.filename)
    elif args.action == "ask":
        if not args.question:
            print(json.dumps({"error": "--question required for ask action"}, ensure_ascii=True))
            sys.exit(1)
        action_ask(args.question, args.backend, args.allow_full_context)
    elif args.action == "status":
        action_status()
    elif args.action == "clear":
        action_clear()
    elif args.action == "models":
        action_models()
