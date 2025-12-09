#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Smoke import test (Updated for Yamabiko)
- Verifies all imports actually used across modules (attributes included)
- Minimal instantiation checks for some libs (no network)
- Validates product_terms.json
- Prints package versions

Run:
  uv run python -m sp_chat_ai_core_yamabiko.smoke_import_test
  # または仮想環境に入って:
  python -m sp_chat_ai_core_yamabiko.smoke_import_test
"""

from __future__ import annotations
import importlib
import json
import sys
from pathlib import Path

# パッケージ名を現在のプロジェクト名に合わせて定義
PKG_NAME = "sp_chat_ai_core_yamabiko"

print(f"🔍 Import smoke test for [{PKG_NAME}] started...\n")

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
    # External Libs
    "openai",
    "google.genai",
    "google.genai.types",
    "langgraph.graph",
    "langchain_core.messages",
    "google.cloud.secretmanager",
    "google.cloud.bigquery",
    "google.cloud.firestore",
    "google.cloud.spanner_v1",
    "janome.tokenizer",
    "rank_bm25",
    "pandas",
    "numpy",
    "pydantic",

    # Internal Modules (Project Specific)
    f"{PKG_NAME}.chat_engine_adk_bq",
    f"{PKG_NAME}.retriever_adk_bq",
    f"{PKG_NAME}.chat_memory",
    f"{PKG_NAME}.firestore_memory_yamabiko",  # 今回修正した箇所
    f"{PKG_NAME}.google_secret_manager",
    f"{PKG_NAME}.support_operation_agent",     # Agent本体
]

for mod in module_targets:
    check_module(mod)

# ========== 2) Attribute-level checks ==========
attr_checks = {
    # typing / standard
    "typing": ["TypedDict", "Annotated", "List", "Sequence", "Dict"],

    # langgraph / langchain
    "langgraph.graph": ["StateGraph", "END"],
    "langchain_core.messages": ["BaseMessage", "HumanMessage", "AIMessage"],

    # Internal Attributes
    f"{PKG_NAME}.chat_engine_adk_bq": ["AdkChatbot"],
    f"{PKG_NAME}.retriever_adk_bq": ["RefactoredRetriever"], # クラス名が異なる場合は修正してください
    f"{PKG_NAME}.chat_memory": ["BaseMemory"],
    f"{PKG_NAME}.firestore_memory_yamabiko": ["FirestoreMemory"], # エラーログから推測されるクラス名
    f"{PKG_NAME}.google_secret_manager": [],
    # AgentはTopLevelでの実行コードが含まれる場合があるため、属性チェックは必須ではないが、import自体は上で確認済み
}

for mod, attrs in attr_checks.items():
    check_attrs(mod, attrs)

# ========== 3) Minimal instantiation checks (No Network) ==========
try:
    from janome.tokenizer import Tokenizer
    _t = Tokenizer()
    toks = [t.surface for t in _t.tokenize("スモークテスト実行中")]
    assert isinstance(toks, list)
    print("✅ janome.Tokenizer minimal tokenize OK")
except Exception as e:
    print(f"❌ janome.Tokenizer failed: {e}")
    record(False, "janome.Tokenizer()", e)

try:
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([["test", "start"], ["smoke", "check"]])
    _ = bm25.get_scores(["check"])
    print("✅ rank_bm25.BM25Okapi minimal init OK")
except Exception as e:
    print(f"❌ rank_bm25.BM25Okapi failed: {e}")
    record(False, "rank_bm25.BM25Okapi()", e)

# ========== 4) Local resource validation ==========
# このファイル自身(__file__)と同じ階層にある product_terms.json を探しに行きます
repo_root = Path(__file__).resolve().parent
json_path = repo_root / "product_terms.json"

if json_path.exists():
    try:
        text = json_path.read_text(encoding="utf-8")
        data = json.loads(text)
        print(f"✅ product_terms.json loaded OK (len={len(data) if hasattr(data,'__len__') else 'n/a'})")
    except Exception as e:
        print(f"❌ product_terms.json validation failed: {e}")
        record(False, "product_terms.json", e)
else:
    print(f"⚠️ product_terms.json not found at {json_path}")
    # 必須ファイルなら record(False, ...) に変更してください

# ========== 5) Versions ==========
print("\n📦 Versions:")
pkgs_to_check = [
    "google-genai", "langgraph", "langchain-core",
    "pandas", "numpy", "rank-bm25", "janome", "pydantic",
    # 内部モジュール自体のバージョンがあれば
]
for pkg in pkgs_to_check:
    ver = get_version(pkg)
    if ver:
        print(f"   - {pkg: <15} : {ver}")

# ========== 6) Summary ==========
print("\n" + "=" * 60)
if failures:
    print(f"❌ NG: {len(failures)} checks failed")
    for name, err in failures:
        print(f"  - {name}: {err}")
    sys.exit(1)
else:
    print("✅ All required imports & checks succeeded. System is ready! 🎉")