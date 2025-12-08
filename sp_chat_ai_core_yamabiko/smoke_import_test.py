#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Smoke import test (extended)
- Verifies all imports actually used across modules (attributes included)
- Minimal instantiation checks for some libs (no network)
- Validates product_terms.json
- Prints package versions

Run:
  uv run python -m sp_chat_ai_core.smoke_import_test
  uv run --active python -m sp_chat_ai_core.smoke_import_test
"""

from __future__ import annotations
import importlib
import json
import sys
from pathlib import Path

print("🔍 Import smoke test (extended) started...\n")

failures: list[tuple[str, BaseException]] = []

def record(ok: bool, name: str, err: BaseException | None):
    if not ok:
        failures.append((name, err))

def check_module(mod: str) -> None:
    try:
        importlib.import_module(mod)
        print(f"✅ {mod} OK")
    except Exception as e:
        print(f"❌ {mod} failed: {e}")
        record(False, mod, e)

def check_attrs(mod: str, attrs: list[str]) -> None:
    try:
        m = importlib.import_module(mod)
    except Exception as e:
        print(f"❌ {mod} import failed (skip attrs): {e}")
        record(False, mod, e)
        return
    for a in attrs:
        try:
            getattr(m, a)
            print(f"   └─ ✅ {mod}.{a} OK")
        except Exception as e:
            print(f"   └─ ❌ {mod}.{a} failed: {e}")
            record(False, f"{mod}.{a}", e)

def get_version(mod_name: str) -> str | None:
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version(mod_name)
        except PackageNotFoundError:
            pass
    except Exception:
        pass
    try:
        m = importlib.import_module(mod_name)
        return getattr(m, "__version__", None) or getattr(m, "VERSION", None)
    except Exception:
        return None

# ========== 1) Modules: presence ==========
module_targets = [
    # stdlib-like are implicitly present; we focus on third-party & internal
    "openai",
    "google.genai",
    "google.genai.types",
    "langgraph.graph",
    "langchain_core.messages",
    "google.cloud.secretmanager",
    "google.cloud.bigquery",
    "google.cloud.firestore",
    "google.cloud.spanner_v1",
    "google.cloud.spanner_v1.param_types",
    "janome.tokenizer",
    "rank_bm25",
    "pandas",
    "numpy",
    "pydantic",

    # internal package modules
    "sp_chat_ai_core.chat_engine_adk_bq",
    "sp_chat_ai_core.retriever_adk_bq",
    "sp_chat_ai_core.chat_memory",
    "sp_chat_ai_core.google_secret_manager",
]
for mod in module_targets:
    check_module(mod)

# ========== 2) Attribute-level checks (reflecting your actual imports) ==========
attr_checks = {
    # typing (stdlib) — confirm symbols exist (optional but explicit)
    "typing": ["TypedDict", "Annotated", "List", "Sequence", "Dict", "Any", "Optional"],

    # langgraph
    "langgraph.graph": ["StateGraph", "END"],  # add "START" if使っていれば: "START"

    # langchain messages
    "langchain_core.messages": [
        "BaseMessage", "HumanMessage", "AIMessage",
        "messages_to_dict", "messages_from_dict"
    ],

    # LLM
    "openai": ["AzureOpenAI"],
    "google.genai": [],             # module presence is enough
    "google.genai.types": [],

    # GCP
    "google.cloud.secretmanager": [],
    "google.cloud.bigquery": [],
    "google.cloud.firestore": [],
    "google.cloud.spanner_v1": [],
    "google.cloud.spanner_v1.param_types": [],

    # NLP / Math
    "janome.tokenizer": ["Tokenizer"],
    "rank_bm25": ["BM25Okapi"],
    "pydantic": ["BaseModel", "Field"],  # "EmailStr" を使うなら追加
    "pandas": [], "numpy": [],

    # internal attributes
    "sp_chat_ai_core.chat_engine_adk_bq": ["AdkChatbot"],
    "sp_chat_ai_core.retriever_adk_bq": ["RefactoredRetriever"],
    "sp_chat_ai_core.chat_memory": ["BaseMemory"],
    "sp_chat_ai_core.google_secret_manager": [],  # as gsm
}
for mod, attrs in attr_checks.items():
    check_attrs(mod, attrs)

# ========== 3) Minimal instantiation checks (no network) ==========
try:
    from janome.tokenizer import Tokenizer
    _t = Tokenizer()
    toks = [t.surface for t in _t.tokenize("これはスモークテストです。")]
    assert isinstance(toks, list)
    print("✅ janome.Tokenizer minimal tokenize OK")
except Exception as e:
    print(f"❌ janome.Tokenizer failed: {e}")
    record(False, "janome.Tokenizer()", e)

try:
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([["これは","テスト"],["BM25","スモーク"]])
    _ = bm25.get_scores(["テスト"])
    print("✅ rank_bm25.BM25Okapi minimal init OK")
except Exception as e:
    print(f"❌ rank_bm25.BM25Okapi failed: {e}")
    record(False, "rank_bm25.BM25Okapi()", e)

try:
    import numpy as np, pandas as pd
    df = pd.DataFrame({"x": np.array([1,2,3], dtype=np.int32)})
    assert df["x"].sum() == 6
    print("✅ pandas/numpy minimal ops OK")
except Exception as e:
    print(f"❌ pandas/numpy minimal ops failed: {e}")
    record(False, "pandas/numpy minimal ops", e)

# ========== 4) Local resource validation ==========
repo_root = Path(__file__).resolve().parent
json_path = repo_root / "product_terms.json"
try:
    text = json_path.read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, (dict, list)):
        raise ValueError("product_terms.json top-level must be dict or list")
    print(f"✅ product_terms.json loaded OK (len={len(data) if hasattr(data,'__len__') else 'n/a'})")
except Exception as e:
    print(f"❌ product_terms.json validation failed: {e}")
    record(False, "product_terms.json", e)

# ========== 5) Versions ==========
print("\n📦 Versions:")
for pkg in [
    "google-genai", "google.genai",
    "openai", "langgraph", "langchain-core",
    "pandas", "numpy", "rank-bm25", "janome", "pydantic",
]:
    ver = get_version(pkg)
    if ver:
        print(f"   - {pkg} version: {ver}")

# ========== 6) Summary ==========
print("\n" + "=" * 60)
if failures:
    print(f"❌ NG: {len(failures)} checks")
    for name, err in failures:
        print(f"  - {name}: {err}")
    sys.exit(1)
else:
    print("✅ All required imports & attribute checks succeeded 🎉")
