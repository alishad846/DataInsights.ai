"""
rag_engine.py — RAG Engine (Node.js integration version)
=========================================================
Self-routing LLM engine that decides CODE vs DIRECT answers.
Uses FAISS for retrieval, Ollama/HuggingFace for LLM.

Called by Node.js backend via rag_cli.py CLI.
"""

import os, json, re, sys
import pandas as pd
import numpy as np

# Optional imports with fallback
try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer

    HAS_ST = True
except ImportError:
    HAS_ST = False

try:
    import pdfplumber

    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    import chardet

    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False

import logging

logger = logging.getLogger("rag_engine")
logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), "rag_data")


class RAGEngine:
    def __init__(self):
        self.device = self._get_device()
        self.embedder = None
        self.dim = 384
        self.index = None
        self.chunks = []
        self.doc_meta = {}
        self._loaded = False
        self._df = None
        self._df_schema = ""
        self.full_context_permission = "full_dataset_ollama"
        self.full_context_char_limit = 60000

    def _get_device(self):
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def _get_embedder(self):
        if self.embedder is None:
            if not HAS_ST:
                raise ImportError(
                    "sentence-transformers not installed. Run: pip install sentence-transformers"
                )
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        return self.embedder

    # ── Document loading ──────────────────────────────────
    def load_document(self, filepath, filename):
        ext = filename.rsplit(".", 1)[-1].lower()
        loaders = {
            "csv": self._load_csv,
            "xlsx": self._load_xlsx,
            "xls": self._load_xlsx,
            "pdf": self._load_pdf,
            "txt": self._load_txt,
            "json": self._load_json,
        }
        if ext not in loaders:
            raise ValueError(f"Unsupported file type: {ext}")

        chunks, meta = loaders[ext](filepath, filename)
        self._build_index(chunks)
        self.doc_meta = meta
        self._loaded = True
        return {**meta, "chunks": len(chunks), "status": "ok"}

    def _load_csv(self, path, name):
        df = pd.read_csv(path)
        for col in df.columns:
            if any(x in col.lower() for x in ["date", "time", "month", "year"]):
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except:
                    pass
        self._df = df
        self._df_schema = self._build_schema(df, name)
        meta = {
            "filename": name,
            "type": "csv",
            "rows": len(df),
            "columns": list(df.columns),
            "preview": df.head(5).to_dict(orient="records"),
        }
        return self._df_to_chunks(df, name), meta

    def _load_xlsx(self, path, name):
        df = pd.read_excel(path)
        self._df = df
        self._df_schema = self._build_schema(df, name)
        meta = {
            "filename": name,
            "type": "xlsx",
            "rows": len(df),
            "columns": list(df.columns),
            "preview": df.head(5).to_dict(orient="records"),
        }
        return self._df_to_chunks(df, name), meta

    def _build_schema(self, df, name):
        lines = [
            f"DataFrame: df  |  File: {name}  |  {len(df)} rows x {len(df.columns)} columns",
            "",
            "Columns:",
        ]
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample = df[col].dropna().head(3).tolist()
            sample_str = ", ".join(str(v) for v in sample)
            lines.append(f"  '{col}' ({dtype}) — e.g. {sample_str}")

        cat_cols = df.select_dtypes(include="object").columns.tolist()
        if cat_cols:
            lines.append("\nUnique values per categorical column:")
            for col in cat_cols[:10]:
                uv = df[col].dropna().unique().tolist()
                uv_str = ", ".join(f"'{v}'" for v in uv[:20])
                lines.append(f"  '{col}': [{uv_str}{'...' if len(uv) > 20 else ''}]")

        num_cols = df.select_dtypes(include="number").columns.tolist()
        if num_cols:
            lines.append("\nNumeric column ranges:")
            for col in num_cols:
                lines.append(
                    f"  '{col}': min={df[col].min():,.2f}, max={df[col].max():,.2f}, mean={df[col].mean():,.2f}"
                )

        return "\n".join(lines)

    def _df_to_chunks(self, df, name):
        chunks = [self._df_schema]
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        if num_cols:
            chunks.append(f"Stats:\n{df[num_cols].describe().round(2).to_string()}")
        for col in cat_cols[:8]:
            chunks.append(
                f"Value counts '{col}':\n{df[col].value_counts().head(15).to_string()}"
            )
        for i in range(0, len(df), 30):
            chunks.append(df.iloc[i : i + 30].to_string(index=False))
        return chunks

    def _load_pdf(self, path, name):
        if not HAS_PDF:
            raise ImportError("pdfplumber not installed. Run: pip install pdfplumber")
        self._df = None
        self._df_schema = ""
        pages = []
        with pdfplumber.open(path) as pdf:
            for i, p in enumerate(pdf.pages):
                t = p.extract_text() or ""
                if t.strip():
                    pages.append(f"[Page {i + 1}]\n{t}")
        full = "\n\n".join(pages)
        return self._split_text(full), {
            "filename": name,
            "type": "pdf",
            "pages": len(pages),
            "preview": full[:500],
        }

    def _load_txt(self, path, name):
        self._df = None
        self._df_schema = ""
        with open(path, "rb") as f:
            raw = f.read()
        enc = "utf-8"
        if HAS_CHARDET:
            enc = chardet.detect(raw)["encoding"] or "utf-8"
        text = raw.decode(enc, errors="replace")
        return self._split_text(text), {
            "filename": name,
            "type": "txt",
            "chars": len(text),
            "preview": text[:500],
        }

    def _load_json(self, path, name):
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            try:
                df = pd.json_normalize(data)
                self._df = df
                self._df_schema = self._build_schema(df, name)
                return self._df_to_chunks(df, name), {
                    "filename": name,
                    "type": "json",
                    "rows": len(df),
                    "columns": list(df.columns),
                    "preview": df.head(5).to_dict(orient="records"),
                }
            except:
                pass
        self._df = None
        self._df_schema = ""
        text = json.dumps(data, indent=2)
        return self._split_text(text), {
            "filename": name,
            "type": "json",
            "preview": text[:500],
        }

    def _split_text(self, text, chunk_size=400, overlap=60):
        words = text.split()
        out = []
        for i in range(0, len(words), chunk_size - overlap):
            c = " ".join(words[i : i + chunk_size])
            if c.strip():
                out.append(c)
        return out or [text[:chunk_size]]

    # ── FAISS Index ───────────────────────────────────────
    def _build_index(self, chunks):
        if not HAS_FAISS:
            self.chunks = chunks
            return
        self.chunks = chunks
        embedder = self._get_embedder()
        emb = embedder.encode(chunks, show_progress_bar=False, batch_size=64)
        emb = np.array(emb, dtype="float32")
        faiss.normalize_L2(emb)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(emb)

    def _retrieve(self, query, top_k=5):
        if not HAS_FAISS or self.index is None:
            return [(0.0, c) for c in self.chunks[:top_k]]
        embedder = self._get_embedder()
        q = np.array(embedder.encode([query]), dtype="float32")
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, top_k)
        return [
            (float(s), self.chunks[i]) for s, i in zip(scores[0], idxs[0]) if i >= 0
        ]

    # ── Safe Code Execution ───────────────────────────────
    def _execute_code(self, code, df):
        safe_builtins = {
            "len": len,
            "sum": sum,
            "min": min,
            "max": max,
            "round": round,
            "abs": abs,
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "list": list,
            "dict": dict,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "sorted": sorted,
            "print": print,
            "isinstance": isinstance,
            "type": type,
            "any": any,
            "all": all,
        }
        local_vars = {"df": df, "pd": pd, "np": np}
        exec_globals = {"__builtins__": safe_builtins}
        try:
            exec(code, exec_globals, local_vars)
            result = local_vars.get("result", None)
            if result is None:
                lines = [l.strip() for l in code.strip().split("\n") if l.strip()]
                if lines:
                    result = eval(lines[-1], exec_globals, local_vars)
            return result, None
        except Exception as e:
            return None, str(e)

    def _result_to_str(self, result):
        if result is None:
            return "No result"
        if isinstance(result, pd.DataFrame):
            return result.head(30).to_string() + (
                f"\n... ({len(result)} rows total)" if len(result) > 30 else ""
            )
        if isinstance(result, pd.Series):
            return result.head(30).to_string() + (
                f"\n... ({len(result)} items)" if len(result) > 30 else ""
            )
        if isinstance(result, (int, np.integer)):
            return f"{result:,}"
        if isinstance(result, (float, np.floating)):
            return f"{result:,.4f}"
        if isinstance(result, list):
            preview = result[:50]
            return str(preview) + (
                f"... ({len(result)} total)" if len(result) > 50 else ""
            )
        return str(result)

    # ── Main Ask ──────────────────────────────────────────
    def ask(self, question, backend="ollama", allow_full_context=False):
        advanced = self._needs_full_context(question)
        has_context = self._df is not None or bool(self.chunks)

        if backend == "ollama" and advanced and has_context and not allow_full_context:
            return self._permission_request(question)

        if allow_full_context and backend == "ollama" and advanced:
            answer = self._ask_full_context(question, backend)
            return self._wrap_answer(answer, backend, intent="full_context")

        if self._df is not None:
            answer = self._ask_smart(question, backend)
            return self._wrap_answer(answer, backend, intent="dataset_qna")

        answer = self._ask_rag(question, backend)
        return self._wrap_answer(answer, backend, intent="rag")

    def _wrap_answer(self, answer, backend, intent):
        if isinstance(answer, dict):
            answer.setdefault("backend", backend)
            answer.setdefault("intent", intent)
            answer.setdefault("source", "rag-engine")
            answer.setdefault("success", True)
            return answer
        return {
            "success": True,
            "answer": answer,
            "backend": backend,
            "intent": intent,
            "source": "rag-engine",
            "confidence": 0.9,
        }

    def _permission_request(self, question):
        return {
            "success": False,
            "answer": (
                "I can give a more concise answer if you allow me to use the "
                "full dataset context with Ollama. Approve full-context analysis?"
            ),
            "intent": "full_context_permission",
            "confidence": 0.99,
            "require_permission": self.full_context_permission,
            "suggested_questions": [
                "Allow full-context analysis",
                "Keep summary mode",
            ],
        }

    def _needs_full_context(self, question):
        q = (question or "").lower()
        keywords = [
            "why", "how should", "how can", "recommend", "strategy", "improve",
            "optimize", "summary", "insight", "forecast", "trend", "compare",
            "comparison", "anomaly", "driver", "correlation", "cause", "impact",
            "best performer", "worst performer", "underperform", "opportunity",
            "what should i do", "what can i do",
        ]
        return any(token in q for token in keywords)

    def _build_full_context(self):
        if self._df is not None:
            parts = [self._df_schema]
            if len(self._df) <= 200:
                parts.append("FULL DATASET:\n" + self._df.to_string(index=False))
            else:
                num_df = self._df.select_dtypes(include="number")
                cat_cols = self._df.select_dtypes(include="object").columns.tolist()

                if not num_df.empty:
                    parts.append("NUMERIC SUMMARY:\n" + num_df.describe().round(2).to_string())

                for col in cat_cols[:8]:
                    vc = self._df[col].value_counts().head(15).to_string()
                    parts.append(f"VALUE COUNTS FOR '{col}':\n{vc}")

                parts.append("SAMPLE ROWS:\n" + self._df.head(30).to_string(index=False))
                parts.append("TAIL ROWS:\n" + self._df.tail(30).to_string(index=False))

            context = "\n\n---\n\n".join(part for part in parts if part)
        else:
            context = "\n\n---\n\n".join(self.chunks)

        if len(context) > self.full_context_char_limit:
            context = (
                context[: self.full_context_char_limit]
                + "\n\n[Context truncated to fit model limits.]"
            )
        return context

    def _ask_full_context(self, question, backend):
        context = self._build_full_context()
        prompt = (
            "You are a concise executive data analyst.\n"
            "The user approved using the full dataset context.\n"
            "Use the information below to answer with the shortest useful answer.\n"
            "Rules:\n"
            "- Be direct and practical.\n"
            "- Use at most 5 short bullets.\n"
            "- Mention the best performer, worst performer, main factor, and one action if relevant.\n"
            "- Do not mention permissions, context size, or internal reasoning.\n\n"
            f"DATASET CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            "ANSWER:"
        )
        if backend == "ollama":
            return self._call_llm(prompt, backend, max_tokens=260)
        return self._ask_rag(question, backend)

    def _ask_smart(self, question, backend):
        prompt = f"""You are a data analyst assistant. A dataset is loaded as `df`.

SCHEMA:
{self._df_schema}

STRICT RULES:
1. Any question with a NUMBER answer -> must use CODE. NEVER guess numbers.
2. DIRECT only for: greetings, column definitions, general explanations (no numbers).
3. If your DIRECT answer would contain a number -> use CODE instead.

FORMAT -- pick one:

CODE:
result = <pandas expression>
EXPLAIN: <one line>

or

DIRECT: <plain English, zero numbers>

EXAMPLES:
"how many female" -> CODE:\nresult = len(df[df['Gender']=='Female'])\nEXPLAIN: Count of females
"average income" -> CODE:\nresult = df['ApplicantIncome'].mean().round(2)\nEXPLAIN: Average income
"top 5 income" -> CODE:\nresult = df.nlargest(5,'ApplicantIncome')[['Loan_ID','ApplicantIncome']]\nEXPLAIN: Top 5
"hi" -> DIRECT: Hello! Ask me anything about your dataset.

QUESTION: {question}
RESPONSE:"""
        raw = self._call_llm(prompt, backend, max_tokens=300)

        # Parse CODE path
        if "CODE:" in raw:
            code_match = re.search(r"CODE:\s*(.*?)(?:EXPLAIN:|$)", raw, re.DOTALL)
            explain_match = re.search(r"EXPLAIN:\s*(.*?)$", raw, re.DOTALL)

            if code_match:
                code = code_match.group(1).strip()
                code = re.sub(r"```python|```", "", code).strip()

                result, error = self._execute_code(code, self._df)

                if error:
                    fix_prompt = f"""This Pandas code failed:
Code: {code}
Error: {error}

Dataset schema:
{self._df_schema}

Write corrected code only (result = ...):"""
                    fixed_code = self._call_llm(fix_prompt, backend, max_tokens=150)
                    fixed_code = re.sub(
                        r"```python|```|CODE:|EXPLAIN:.*", "", fixed_code
                    ).strip()
                    result, error2 = self._execute_code(fixed_code, self._df)
                    if error2:
                        return self._ask_rag(question, backend)

                result_str = self._result_to_str(result)

                if isinstance(result, (pd.DataFrame, pd.Series)):
                    rows = len(result)
                    header = f"Here are the results ({rows} {'row' if rows == 1 else 'rows'}):\n\n"
                    return header + result_str

                if isinstance(result, (int, float, np.integer, np.floating)):
                    hint = explain_match.group(1).strip() if explain_match else ""
                    explain_prompt = f"""A user asked: "{question}"
The exact answer computed from the data is: {result_str}
{f"({hint})" if hint else ""}
Write ONE clear sentence stating this answer. Include the number. No extra text."""
                    return self._call_llm(explain_prompt, backend, max_tokens=80)

                hint = explain_match.group(1).strip() if explain_match else ""
                explain_prompt = f"""A user asked: "{question}"

Exact result from the dataset:
{result_str}

{f"Context: {hint}" if hint else ""}

Give a clear, concise answer using these exact values. No disclaimers."""
                return self._call_llm(explain_prompt, backend, max_tokens=200)

        # Parse DIRECT path
        if "DIRECT:" in raw:
            direct_match = re.search(r"DIRECT:\s*(.*?)$", raw, re.DOTALL)
            if direct_match:
                return direct_match.group(1).strip()

        cleaned = re.sub(r"CODE:|DIRECT:|EXPLAIN:", "", raw).strip()
        return cleaned if cleaned else self._ask_rag(question, backend)

    def _ask_rag(self, question, backend):
        hits = self._retrieve(question, top_k=5)
        context = "\n\n---\n\n".join(c for _, c in hits)
        if self._df_schema:
            context = self._df_schema + "\n\n---\n\n" + context
        prompt = (
            f"You are a helpful data analyst. Answer using ONLY the context.\n"
            f"Be concise and direct.\n\nCONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\nANSWER:"
        )
        return self._call_llm(prompt, backend, max_tokens=350)

    # ── LLM Caller ───────────────────────────────────────
    def _call_llm(self, prompt, backend, max_tokens=350):
        result = None
        if backend == "ollama":
            result = self._ask_ollama(prompt, max_tokens)
        elif backend == "huggingface":
            result = self._ask_hf(prompt, max_tokens)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        if result is None:
            return (
                "AI temporarily unavailable. Please try again or use a different model."
            )

        return result

    def _ask_ollama(self, prompt, max_tokens=350) -> str | None:
        import requests

        preferred = [
            "codellama:7b",
            "mistral",
            "mistral:latest",
            "llama3.2",
            "llama3.2:latest",
            "llama3",
            "gemma2:2b",
            "phi3:mini",
            "tinyllama",
        ]
        try:
            available = [
                m["name"]
                for m in requests.get("http://localhost:11434/api/tags", timeout=5)
                .json()
                .get("models", [])
            ]
        except:
            available = []
        model = next(
            (p for p in preferred if p in available),
            available[0] if available else "mistral",
        )
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": max_tokens,
                        "num_gpu": 99,
                    },
                },
                timeout=300,
            )
            resp.raise_for_status()
            response = resp.json().get("response", "").strip()

            if not response:
                logger.warning("[RAG] Empty response from Ollama")
                return None

            error_patterns = [
                "does not support image",
                "cannot read image",
                "not support",
                "vision",
                "multimodal",
            ]
            if any(pattern in response.lower() for pattern in error_patterns):
                logger.warning(
                    "[RAG] Model does not support image input - using fallback"
                )
                return None

            return response
        except requests.exceptions.ConnectionError:
            logger.warning("[RAG] Ollama not running")
            return None
        except requests.exceptions.Timeout:
            logger.warning("[RAG] Ollama timed out")
            return None
        except Exception as e:
            logger.warning(f"[RAG] Ollama error: {e}")
            return None

    def _ask_hf(self, prompt, max_tokens=350):
        try:
            from transformers import pipeline as hf_pipeline
            import torch

            model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            pipe = hf_pipeline(
                "text-generation",
                model=model_id,
                device_map="auto",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                max_new_tokens=max_tokens,
                do_sample=False,
            )
            msgs = [
                {
                    "role": "system",
                    "content": "You are a Python/Pandas expert and data analyst.",
                },
                {"role": "user", "content": prompt},
            ]
            out = pipe(msgs)
            text = out[0]["generated_text"]
            if isinstance(text, list):
                for m in reversed(text):
                    if m.get("role") == "assistant":
                        return m["content"].strip()
            return str(text).strip()
        except Exception as e:
            return f"HuggingFace error: {e}"

    def is_loaded(self):
        return self._loaded

    def clear(self):
        self.index = None
        self.chunks = []
        self.doc_meta = {}
        self._loaded = False
        self._df = None
        self._df_schema = ""

    def get_status(self):
        ollama_ok = False
        try:
            import requests

            ollama_ok = (
                requests.get("http://localhost:11434/api/tags", timeout=2).status_code
                == 200
            )
        except:
            pass
        hf_ok = False
        try:
            import transformers

            hf_ok = True
        except:
            pass
        return {
            "ollama": ollama_ok,
            "huggingface": hf_ok,
            "dataset_loaded": self._loaded,
            "doc_meta": self.doc_meta,
            "device": self.device,
            "has_faiss": HAS_FAISS,
            "has_sentence_transformers": HAS_ST,
        }
