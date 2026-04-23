"""
cognitive_engine.py — Intelligent LLM-based Dataset Q&A Engine
=========================================================
Uses Groq (llama-3.3-70b-versatile) to translate natural language questions into
Pandas code, executes the code safely using AST Sandboxing, and generates a
natural language response.

Fallback policy (query_engine.py is called ONLY when appropriate):
  - LLM_UNAVAILABLE  : Groq API key missing or network/auth error
  - EXEC_FAILED      : Generated code raised an exception in the sandbox
  - SCHEMA_MISMATCH  : LLM explicitly set result = "IMPOSSIBLE"
  - NOT used for     : conceptual/reasoning questions (anomaly, insight, summary,
                       forecast, comparison) — those return a clean service-error
                       message instead of a misleading canned fallback answer.
"""

import argparse
import ast
import json
import logging
import os
import re
import sys
import time

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

import numpy as np
import pandas as pd
import warnings

try:
    from groq import Groq
    GROQ_IMPORT_ERROR = None
except ImportError as exc:
    Groq = None
    GROQ_IMPORT_ERROR = exc

warnings.filterwarnings("ignore")
load_dotenv()

# ─── Fallback engine import ───────────────────────────────────────────────────
try:
    from pipeline.query_engine import (
        DatasetArtifacts,
        answer_question as fallback_answer,
        detect_intent as fallback_detect_intent,
    )
except ImportError:
    from query_engine import (
        DatasetArtifacts,
        answer_question as fallback_answer,
        detect_intent as fallback_detect_intent,
    )

logger = logging.getLogger("cognitive_engine")
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s [%(name)s]: %(message)s",
    stream=sys.stderr,
)

# ─── Config ───────────────────────────────────────────────────────────────────

API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_IMPORT_ERROR:
    logger.info("groq package is not installed. Falling back to query_engine only.")

if API_KEY and Groq is not None:
    client = Groq(api_key=API_KEY)
else:
    if not API_KEY:
        logger.info("GROQ_API_KEY is not set in the environment.")
    elif Groq is None:
        logger.info("groq package is unavailable; LLM calls will use fallback handling.")
    client = None

# llama-3.3-70b-versatile: 128K context, actively maintained.
# Replaces the deprecated llama3-70b-8192 (8K context).
GROQ_MODEL = "llama-3.3-70b-versatile"

# ─── Fallback routing ─────────────────────────────────────────────────────────

# The fallback engine can now answer most dataset questions directly, including
# advanced dataset requests. Keep this set empty unless we add a truly
# unsupported intent in the future.
REASONING_ONLY_INTENTS = set()

def _is_reasoning_question(question: str) -> bool:
    """Returns True if the question needs LLM reasoning and the fallback cannot help."""
    try:
        intent = fallback_detect_intent(question)
        return intent in REASONING_ONLY_INTENTS
    except Exception:
        return False


def _run_fallback(question: str, artifacts: DatasetArtifacts, reason: str) -> dict:
    """
    Attempts the regex-based fallback engine with a documented reason.

    - If the question is reasoning-only, skips fallback entirely and returns a
      clean "service unavailable" message. The fallback engine would only return
      a misleading canned response for these (e.g. "No anomaly analysis available").
    - Otherwise calls fallback_answer and tags the result with fallback_reason
      so the caller/Node.js can log or display why fallback was used.

    reason: human-readable string, one of:
        LLM_UNAVAILABLE   — Groq API call failed
        EXEC_FAILED       — sandbox raised an exception
        SCHEMA_MISMATCH   — LLM set result = "IMPOSSIBLE"
    """
    if _is_reasoning_question(question):
        logger.info(f"Fallback skipped for reasoning question. Reason: {reason}")
        return {
            "success": False,
            "answer": (
                "I'm having trouble connecting to the AI service right now. "
                "Please try again in a moment."
            ),
            "intent": "error",
            "confidence": 0.0,
            "fallback_used": False,
            "fallback_skipped_reason": reason,
        }

    logger.info(f"Falling back to query_engine. Reason: {reason}")
    try:
        fb_result = fallback_answer(question, artifacts)
        fb_result["fallback_used"] = True
        fb_result["fallback_reason"] = reason
        # query_engine doesn't set "success" — normalise it
        if "success" not in fb_result:
            fb_result["success"] = fb_result.get("confidence", 0.0) > 0.0
        return fb_result
    except Exception as fallback_e:
        logger.error(f"Fallback engine also failed: {fallback_e}")
        return {
            "success": False,
            "answer": "I encountered an error processing your question. Please try again.",
            "intent": "error",
            "confidence": 0.0,
            "fallback_used": True,
            "fallback_reason": reason,
            "fallback_error": str(fallback_e),
        }

# ─── AST Sandbox ──────────────────────────────────────────────────────────────

BLOCKED_IMPORTS = {
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "requests", "httpx",
    "urllib", "ftplib", "smtplib", "http", "importlib", "runpy", "pty", "signal",
    "ctypes", "cffi", "pickle", "shelve", "multiprocessing", "threading", "concurrent"
}
BLOCKED_ATTRS = {
    "__import__", "__builtins__", "__loader__", "__spec__", "__subclasses__",
    "__bases__", "__mro__", "__code__", "__globals__", "__closure__",
    "__reduce__", "__reduce_ex__"
}
BLOCKED_CALLS = {"eval", "exec", "compile", "open", "input", "breakpoint"}


class ASTSafetyChecker(ast.NodeVisitor):
    def visit_Import(self, node):
        for a in node.names:
            if a.name.split(".")[0] in BLOCKED_IMPORTS:
                raise ValueError(f"Import '{a.name}' not allowed.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if (node.module or "").split(".")[0] in BLOCKED_IMPORTS:
            raise ValueError(f"Import from '{node.module}' not allowed.")
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
            raise ValueError(f"Call to '{node.func.id}' not allowed.")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr in BLOCKED_ATTRS:
            raise ValueError(f"Access to '{node.attr}' not allowed.")
        self.generic_visit(node)


# Sentinel: distinguishes "result never assigned" from "result = 0 / False / None / ''"
_SENTINEL = object()


def safe_execute(code: str, df: pd.DataFrame):
    """Executes LLM-generated pandas code safely, returns the `result` variable."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"Syntax error in generated code: {e}")

    ASTSafetyChecker().visit(tree)
    byte_code = compile(tree, "<query>", "exec")

    sb = {
        "len": len, "range": range, "list": list, "dict": dict, "tuple": tuple,
        "set": set, "str": str, "int": int, "float": float, "bool": bool,
        "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
        "sorted": sorted, "enumerate": enumerate, "zip": zip, "map": map,
        "filter": filter, "isinstance": isinstance, "type": type,
        "print": print, "repr": repr, "hasattr": hasattr,
        "True": True, "False": False, "None": None, "any": any, "all": all,
    }
    g  = {"__builtins__": sb, "df": df, "pd": pd, "np": np}
    lv = {"result": _SENTINEL}

    exec(byte_code, g, lv)  # noqa: S102

    # Use sentinel — NOT `if res is not None` — so falsy values like 0, False,
    # "", or an empty DataFrame are returned correctly instead of being swallowed.
    res = lv.get("result", _SENTINEL)
    if res is not _SENTINEL:
        return res
    return None

# ─── LLM Prompts ─────────────────────────────────────────────────────────────

QUERY_SYSTEM_PROMPT = """You are an expert pandas code generator.
Given a dataset schema and a user question, output ONLY raw Python code that:
1. Uses variable `df` (a pandas DataFrame, already loaded and available).
2. Computes the answer or modification and assigns it to a variable called `result`.
3. NEVER IMPORTS anything (no os, sys, math, etc.).
4. CRITICAL: When using boolean indexing in pandas, ALWAYS use `&` or `|` instead
   of Python `and` or `or`. ALWAYS wrap each condition in parentheses,
   e.g. `df[(df['a'] > 1) & (df['b'] < 2)]`.
5. CRITICAL: For modifications, ALWAYS use `df.loc`.
   Example: `df.loc[df['col'] == 'old_val', 'col'] = 'new_val'`.
   Do not use `df.apply` if possible.
6. For modifications, modify `df` directly and set `result = df`.
7. For queries, set `result` to the computed answer (a DataFrame, Series, or scalar).
8. Use ONLY column names present in the dataset schema provided.
Respond with ONLY raw Python code — no markdown blocks, no backticks, no explanations.
If the question cannot be answered with the available schema, set `result = "IMPOSSIBLE"`.
"""

NL_SYSTEM_PROMPT = """You are a helpful and intelligent data analyst chatbot.
You are given a user's question and the raw computational result from a pandas query.
Write a clear, concise, and professional answer in natural language.
- Be specific with numbers. Format large numbers with commas.
- Do NOT show code or raw tabular structures.
- Keep the tone helpful and direct.
"""

# ─── LLM callers ─────────────────────────────────────────────────────────────

def call_llm(system: str, prompt: str) -> str:
    """Calls Groq for pandas code generation (temperature=0 for determinism)."""
    if not client:
        if GROQ_IMPORT_ERROR:
            raise ValueError("groq package is not installed.")
        raise ValueError("GROQ_API_KEY is not set.")
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        model=GROQ_MODEL,
        temperature=0.0,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def call_llm_nl(system: str, prompt: str) -> str:
    """Calls Groq for natural language answer generation (slight creativity allowed)."""
    if not client:
        if GROQ_IMPORT_ERROR:
            raise ValueError("groq package is not installed.")
        raise ValueError("GROQ_API_KEY is not set.")
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        model=GROQ_MODEL,
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()

# ─── Modify intent detection ──────────────────────────────────────────────────

MODIFY_KEYWORDS = [
    "fill", "replace", "update", "set", "delete", "drop", "remove", "rename",
    "insert", "add column", "change", "modify", "clean", "impute", "assign",
    "overwrite", "convert", "map", "encode", "strip", "trim", "lowercase", "uppercase",
]

def is_modify_prompt(q: str) -> bool:
    """
    Returns True if the question is asking to mutate the dataset.
    Uses word-boundary regex so substrings like 'dataset', 'remap', 'timestamp'
    do not produce false positives (e.g. 'set' inside 'dataset').
    """
    q_lower = q.lower()
    if _is_advanced_data_request(q_lower):
        return False
    return any(
        re.search(rf"\b{re.escape(kw)}\b", q_lower)
        for kw in MODIFY_KEYWORDS
    )


def _is_advanced_data_request(question: str) -> bool:
    """Detect requests for deeper analysis or advanced chat, not dataset edits."""
    q = (question or "").lower()
    advanced_patterns = [
        r"\badvanced\s+chat\b",
        r"\badvanced\s+analysis\b",
        r"\badvanced\s+dataset\b",
        r"\badvanced\s+data\s+set\b",
        r"\bfull\s+dataset\b",
        r"\bcomplete\s+dataset\b",
        r"\bfull\s+data\b",
        r"\bcomplete\s+data\b",
        r"\bdeep\s+analysis\b",
        r"\buse\s+the\s+full\s+data\b",
        r"\buse\s+all\s+data\b",
        r"\bmore\s+concise\s+answer\b",
        r"\badvanced\b.*\bdata\b",
    ]
    return any(re.search(pat, q) for pat in advanced_patterns)

# ─── Main Dispatcher ──────────────────────────────────────────────────────────

def process_query_intelligent(
    question: str,
    artifacts: DatasetArtifacts,
    permissions: dict,
) -> dict:
    """
    Full pipeline:
      1. Permission gate
      2. Dataset availability check
      3. LLM → pandas code generation
      4. AST sandbox execution
      5. Natural language answer generation

    Fallback to query_engine.py is attempted only when:
      - Groq API call fails          (reason: LLM_UNAVAILABLE)
      - Sandbox execution fails      (reason: EXEC_FAILED)
      - LLM returns "IMPOSSIBLE"     (reason: SCHEMA_MISMATCH)

    For questions that require reasoning (anomaly, insight, summary, forecast,
    comparison), the fallback is skipped and a clean service-error is returned
    instead of the misleading canned responses from query_engine.py.
    """

    # ── 1. Permission gate ────────────────────────────────────────────────────
    modify_intent = is_modify_prompt(question)

    if modify_intent:
        if not permissions.get("can_edit", False):
            return {
                "success": False,
                "require_permission": "can_edit",
                "answer": (
                    "You do not have permission to modify this dataset. "
                    "Would you like to request edit access from an administrator?"
                ),
                "intent": "modify_denied",
                "confidence": 1.0,
                "error": "permission_denied",
            }
    else:
        if not permissions.get("can_query", True) and not permissions.get("can_view", True):
            return {
                "success": False,
                "require_permission": "can_query",
                "answer": "You do not have permission to query this dataset. Would you like to request access?",
                "intent": "query_denied",
                "confidence": 1.0,
                "error": "permission_denied",
            }

    # ── 2. Dataset availability ───────────────────────────────────────────────
    if not artifacts.available or artifacts.df is None:
        return {
            "success": False,
            "answer": "The dataset is still being processed or could not be loaded. Please wait.",
            "intent": "unavailable",
            "confidence": 0.0,
        }

    df     = artifacts.df
    schema = artifacts.schema

    structured_intent = fallback_detect_intent(question)
    advanced_request = _is_advanced_data_request(question)
    if not modify_intent and (structured_intent != "unknown" or advanced_request):
        try:
            structured_result = fallback_answer(question, artifacts)
            if "success" not in structured_result:
                structured_result["success"] = structured_result.get("confidence", 0.0) > 0.0
            structured_result.setdefault("source", "query_engine")
            return structured_result
        except Exception as e:
            logger.error(f"Structured fallback failed: {e}")

    if client is None:
        reason = (
            "LLM_UNAVAILABLE: groq package is not installed."
            if GROQ_IMPORT_ERROR
            else "LLM_UNAVAILABLE: GROQ_API_KEY is not set."
        )
        return _run_fallback(question, artifacts, reason=reason)

    # ── 3. LLM → pandas code generation ──────────────────────────────────────
    op_hint = (
        "MODIFY OPERATION — apply changes to df in-place using df.loc, then set result = df"
        if modify_intent else
        "READ OPERATION — do NOT modify df. Compute the answer and set result = <answer>"
    )
    prompt = (
        f"Dataset schema (JSON):\n{json.dumps(schema, indent=2)}\n\n"
        f"Operation Type: {op_hint}\n\n"
        f"User Question: {question}\n\n"
        "Output ONLY raw Python code. No markdown, no backticks, no explanation."
    )

    t0 = time.time()
    try:
        raw_code = call_llm(QUERY_SYSTEM_PROMPT, prompt)
        # Strip any accidental markdown fences from the LLM response
        raw_code = re.sub(r"^```(?:python)?\s*", "", raw_code, flags=re.MULTILINE)
        raw_code = re.sub(r"\s*```$", "", raw_code, flags=re.MULTILINE).strip()
    except Exception as e:
        logger.error(f"LLM code generation failed: {e}")
        return _run_fallback(question, artifacts, reason=f"LLM_UNAVAILABLE: {e}")

    # ── 4. AST sandbox execution ──────────────────────────────────────────────
    try:
        # Read queries get a df.copy() so the sandbox cannot mutate the live df.
        # Modify operations get the real df so df.loc mutations are preserved.
        df_context = df if modify_intent else df.copy()
        raw_result = safe_execute(raw_code, df_context)
        exec_time  = round(time.time() - t0, 4)
    except Exception as e:
        logger.error(f"Sandbox execution failed: {e}\nGenerated code:\n{raw_code}")
        fb = _run_fallback(question, artifacts, reason=f"EXEC_FAILED: {e}")
        fb["generated_code"] = raw_code   # attach bad code for debugging
        return fb

    # LLM signalled it cannot answer with the current schema
    if isinstance(raw_result, str) and raw_result == "IMPOSSIBLE":
        return _run_fallback(
            question, artifacts,
            reason="SCHEMA_MISMATCH: LLM set result='IMPOSSIBLE'"
        )

    # LLM forgot to assign `result` — treat as execution failure
    if raw_result is None:
        return _run_fallback(
            question, artifacts,
            reason="EXEC_FAILED: Generated code did not assign a value to `result`"
        )

    # ── 5. Result formatting ──────────────────────────────────────────────────
    MAX_ROWS = 100

    if modify_intent and isinstance(raw_result, pd.DataFrame):
        # Sync the in-memory artifact so downstream callers see the updated df
        artifacts.df = raw_result

        if artifacts.csv_file:
            ext = os.path.splitext(artifacts.csv_file)[1].lower()
            try:
                if ext == ".csv":
                    raw_result.to_csv(artifacts.csv_file, index=False)
                elif ext in (".xlsx", ".xls"):
                    raw_result.to_excel(artifacts.csv_file, index=False)
                elif ext == ".json":
                    raw_result.to_json(artifacts.csv_file, orient="records")
            except Exception as e:
                logger.error(f"Failed to persist modified dataset: {e}")

        result_str = (
            f"Dataset updated successfully. "
            f"New shape: {raw_result.shape[0]:,} rows × {raw_result.shape[1]} columns."
        )

    elif isinstance(raw_result, pd.DataFrame):
        result_str = raw_result.head(MAX_ROWS).to_string(index=False)

    elif isinstance(raw_result, pd.Series):
        result_str = raw_result.head(MAX_ROWS).to_string()

    elif isinstance(raw_result, (np.integer, np.floating, int, float)):
        result_str = str(raw_result)

    else:
        result_str = str(raw_result)

    # ── 6. Natural language answer ────────────────────────────────────────────
    nl_prompt = (
        f"User question: {question}\n\n"
        f"Computational Result:\n{result_str[:3000]}\n\n"
        "Write a natural language answer based ONLY on this result."
    )
    try:
        final_answer = call_llm_nl(NL_SYSTEM_PROMPT, nl_prompt)
    except Exception as e:
        logger.error(f"NL generation failed: {e}")
        # NL failure is non-critical — surface the raw result as plain text
        final_answer = f"Here is the result:\n{result_str[:500]}"

    return {
        "success": True,
        "answer": final_answer,
        "intent": "modify" if modify_intent else "insight",
        "confidence": 0.95,
        "generated_code": raw_code,
        "raw_text_result": result_str[:1500],
        "execution_time": exec_time,
    }


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cognitive Q&A Engine (LLM + Pandas)")
    parser.add_argument("--user_id",     required=True)
    parser.add_argument("--dataset_id",  required=True)
    parser.add_argument("--question",    required=True)
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--csv_file",    default=None)
    parser.add_argument(
        "--permissions",
        default='{"can_view":true,"can_query":true,"can_edit":false}',
        help="JSON string of user permissions",
    )
    args = parser.parse_args()

    try:
        permissions = json.loads(args.permissions)
    except json.JSONDecodeError:
        permissions = {"can_view": True, "can_query": True, "can_edit": False}

    try:
        from pipeline.query_engine import resolve_dataset_dir
    except ImportError:
        from query_engine import resolve_dataset_dir

    dataset_dir = args.dataset_dir or resolve_dataset_dir(args.user_id, args.dataset_id)
    artifacts   = DatasetArtifacts(dataset_dir, csv_file=args.csv_file)
    result      = process_query_intelligent(args.question, artifacts, permissions)

    # Single JSON line to stdout — Node.js reads this
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
