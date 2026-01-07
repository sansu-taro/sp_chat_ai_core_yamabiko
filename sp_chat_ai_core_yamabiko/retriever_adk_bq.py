# refactored_retriever.py (修正版)

from __future__ import annotations
import os, re, json
import logging
from typing import List, Set, Dict, Any, Optional

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from rank_bm25 import BM25Okapi
from pathlib import Path

from google.cloud import bigquery
from google.cloud import spanner_v1
from google.cloud.spanner_v1 import param_types
from janome.tokenizer import Tokenizer

# 先頭の import 群に追加
import threading, unicodedata, hashlib

# ====== 定数・環境依存 ======

_thread_local = threading.local()
BIGQUERY_PROJECT_ID = "smarthr-customer-support"
BIGQUERY_DATASET_ID = "sandbox"


# VECTOR_TABLE_BQ = f"{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.knowledges_vector_v2"
# QA_TABLE_BQ     = f"{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.knowledges_qa_part_id_fill"
VECTOR_TABLE_BQ = f"{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.knowledges_vector_v2_202411_202511"
QA_TABLE_BQ     = f"{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.knowledges_qa_part_id_fill_2025"
LOG_TABLE_BQ    = f"{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.retrieval_logs"

SPANNER_PROJECT_ID  = "smarthr-customer-support"
SPANNER_INSTANCE_ID = "smarthr-customer-support"
SPANNER_DATABASE_ID = "knowledge_search"
KNOWLEDGES_TABLE    = "knowledges"
VECTOR_COLUMN_SPN   = "vector"
BODY_COLUMN_SPN     = "body"

# ベクトル/BM25のパラメタ
VEC_TOPK     = 300
BM25_TOPK    = 300
BLEND_ALPHA  = 0.7
BLEND_BETA   = 0.3
KEEP_TOPK    = 10
FINAL_TOPN   = 3  # 呼び出し側で上書き可
JAP_PUNCT = "。、！？”“「」『』（）()［］[]｛｝{}・…：；．，／＼｜〜~"
RE_CTRL   = re.compile(r"[\u0000-\u001F\u007F]")               # 制御文字
RE_URL    = re.compile(r"https?://\S+")
RE_SPACE  = re.compile(r"\s+")
RE_PUNRUN = re.compile(r"([_\-*=~!@#$%^&+|\\/]{3,})")          # 記号連発
RE_FALLBK = re.compile(r"[一-龥々〆ヵヶぁ-んァ-ヶｱ-ﾝﾞﾟーA-Za-z0-9]+")  # 簡易トークナイザ


# ====== Spanner Search Index 設定 ======
SSEARCH_INDEX_NAME = "KnowledgesSearchAll"  # ★非パーティションの新インデックス名
USE_TOKENLIST_CONCAT = True                 # 列横断で検索する


# ストップワード
DEFAULT_STOPWORDS: Set[str] = set(
    ["お世話","の","に","は","を","た","が","で","て","と","し","れ","さ","ある","いる","も",
     "する","から","な","こと","として","い","や","れる","など","なっ","ない","この","ため",
     "その","あっ","よう","また","もの","という","あり","まで","られ","なる","へ","か","だ",
     "これ","によって","により","おり","より","による","ず","なり","られる","において","ば",
     "なれ","き","つ","における","および","いう","さらに","でも","ら","その他","そして",
     "に関する","すなわち","つまり","ただし","しかし","したがって","それでも","その後",
     "そこで","それで","それから","それぞれ","それでは","。","、","「","」","（","）"," ","　",
     "他","職員","テナント","方法","者","タ","お願い","先日","見かけ","大変","現在","上記","先程",
     "対応","内容","案内","感謝","時点","問合せ","拝見","過去","要望","リリース","詳細","申し訳","時間",
     "問題","ギリギリ","回答","間際","連絡","丁寧","有難く","有難く","今後","有難かっ","嬉しく","不明",
     "理解","検討","間違い","是非", "利用", "宜しく","良い","実感","返信","お手数","通り","認識","便利",
     "解決","確認","実現", "感激","承知","小さ","願い事","改善","素晴らしい","お知らせ","お忙しい",
     "弊社","心待ち","対処","質問","伺い","背景","SmartHR","状況","情報","管理","機能",
    ]
)

# トピック辞書（必要に応じて拡張）
# 「主題語: 同義語/関連語（本文マッチ用）」

# 1. このスクリプトファイル（.py）の絶対パスを取得
# .resolve() はシンボリックリンクなどを解決して完全なパスにします
script_path = Path(__file__).resolve()

# 2. このスクリプトファイルが存在するディレクトリのパスを取得
# .parent がディレクトリを指します
script_dir = script_path.parent

# 3. 読み込むJSONファイルの絶対パスを構築
# / 演算子で安全にパスを結合できます
json_filename = "product_terms.json"
pt_path = script_dir / json_filename

# ↓↓↓ 以下は元のコードと同じ ↓↓↓
try:
    # Pathオブジェクトはそのまま open() に渡せます
    with open(pt_path, 'r', encoding='utf-8') as f:
        TOPIC_LEXICON = json.load(f)
except FileNotFoundError:
    print(f"警告: JSONファイル '{pt_path}' が見つかりません。")
    TOPIC_LEXICON = {} # ファイルが見つからない場合のデフォルト値
except json.JSONDecodeError as e:
    print(f"警告: JSONファイル '{pt_path}' のデコードに失敗しました: {e}")
    TOPIC_LEXICON = {}


# ====== ユーティリティ ======
def _minmax(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    mn, mx = float(series.min()), float(series.max())
    if mx - mn < 1e-12:
        return pd.Series(0.0, index=series.index)
    return (series - mn) / (mx - mn)


def _hit_ratio(text: str, terms: List[str]) -> float:
    """terms のうち何語が含まれるかの比率（本文・タイトルの静的判定用）"""
    if not text or not terms:
        return 0.0
    t = text
    hit = sum(1 for w in terms if w and w in t)
    return hit / max(1, len(terms))


def _primary_topic_from_query(query: str) -> Optional[str]:
    """簡易：クエリに最もよく現れる主題語で primary topic を決める"""
    q = query or ""
    best_topic, best_score = None, 0.0
    for topic, syns in TOPIC_LEXICON.items():
        score = _hit_ratio(q, [topic] + syns[:3])  # 主題名 + 代表語で軽く判定
        if score > best_score:
            best_topic, best_score = topic, score
    return best_topic


def _topic_score_row(row: Dict[str, Any], primary_topic: str) -> float:
    """
    静的トピック判定（汎用語彙は使わず、主題語・同義語のみ）
    タイトル/URL/カテゴリ（軽） + 本文（重）で評価。公式ソースに微小バイアス。
    """
    title_side = f"{row.get('title','')} {row.get('url','')} {row.get('category','')}"
    body_side  = row.get("body", "") or ""

    a = _hit_ratio(title_side, [primary_topic])                       # 主題名ヒット（タイトル/カテゴリ）
    b = _hit_ratio(body_side, TOPIC_LEXICON.get(primary_topic, []))   # 本文で同義語ヒット

    # 公式っぽいURLにのみ微小バイアス（断定への悪影響にならない程度）
    source_bias = 0.0
    url = (row.get("url") or "").lower()
    if "support.smarthr.jp" in url:
        source_bias = 0.05

    return 0.5 * a + 0.5 * b + source_bias


def _get_tokenizer() -> Tokenizer:
    tok = getattr(_thread_local, "janome_tok", None)
    if tok is None:
        tok = Tokenizer(mmap=True)  # 各スレッド専用インスタンス
        _thread_local.janome_tok = tok
    return tok


# ====== Retriever ======
class RefactoredRetriever:
    def __init__(self, stopwords: Set[str] | None = None):
        self.logger = logging.getLogger(__name__)
        self.bq_client = bigquery.Client(project=BIGQUERY_PROJECT_ID)

        spanner_client = spanner_v1.Client(project=SPANNER_PROJECT_ID)
        instance = spanner_client.instance(SPANNER_INSTANCE_ID)
        self.spanner_db = instance.database(SPANNER_DATABASE_ID)

        #self.tokenizer = Tokenizer()
        #self.stopwords = stopwords if stopwords is not None else DEFAULT_STOPWORDS
        # self.tokenizer = Tokenizer(mmap=True)  # ← mmap で安定＆高速化
        # self._tok_lock = threading.Lock()      # ← 追加：並行実行ガード
        
        self.stopwords = DEFAULT_STOPWORDS
        self.hiragana_pattern = re.compile(r"^[ぁ-んー]+$")
        # 新しいトークナイザ用の正規表現パターン
        self.RE_SYMBOL = re.compile(r"^[\W_]+$")
        self.RE_ALNUM = re.compile(r"^[0-9A-Za-z]+$")


    # --- Sanitize / Chunk / Fallback ---
    def _sanitize_text(self, text: str) -> str:
        if not text:
            return ""
        t = unicodedata.normalize("NFKC", str(text))
        t = RE_CTRL.sub(" ", t)         # 制御文字→空白
        t = RE_URL.sub(" ", t)          # URL は削る（BM25 的にもノイズ）
        t = RE_PUNRUN.sub(r"\1", t)     # 記号連打の圧縮
        t = RE_SPACE.sub(" ", t).strip()
        MAX_LEN = 10000                 # 極端長文の安全弁
        return t[:MAX_LEN]

    def _chunk_iter(self, text: str, size: int = 3000):
        for i in range(0, len(text), size):
            yield text[i:i+size]

    def _fallback_tokenize(self, chunk: str) -> list[str]:
        return RE_FALLBK.findall(chunk)

    
    def build_search_query(self, query_text: str, max_terms: int = 5) -> str:
        """クエリ→正規化→共通トークナイザ→AND連結。ロジックを一元化。"""
        if not isinstance(query_text, str) or not query_text.strip():
            return ""
        # 全角→半角、句読点→空白
        q = unicodedata.normalize("NFKC", query_text)
        for ch in JAP_PUNCT:
            q = q.replace(ch, " ")
        
        q = re.sub(r"\s+", " ", q).strip()
        
        # 既存の共通トークナイザを適用
        terms = self.tokenize_japanese_with_stopwords(q)
        if not terms:
            return ""
        
        # 重複排除＋max_terms制限
        uniq, seen = [], set()
        for t in terms:
            if t not in seen:
                uniq.append(t)
                seen.add(t)        
        compact = uniq[:max_terms]
        # Spanner SEARCH用に AND 連結（必要に応じて " ".join(compact) に変更可）
        #re_token = " | ".join(compact)
        re_token = " AND ".join(compact)
        
        return re_token


    # ---------- Public API ----------
    def search_all_sources(
        self,
        query_text: str,
        query_vector: np.ndarray,
        top_n: int = 5,
        session_id: Optional[str] = None,
        message_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        self.logger.info(f"検索開始: query='{query_text}', session_id='{session_id}'")
        # 1) Spanner: 全文 × ベクトル（非パーティション index）
        # sp_df = self._search_knowledge_fulltext_and_vector(
        #     query_text=query_text,
        #     query_vector=query_vector,
        #     top_n=60,
        #     alpha_vec=0.60, beta_ft=0.40
        # )
        
        # if sp_df is None or sp_df.empty:
        #     # フォールバック：従来のベクトル→BM25ブレンド
        #     fb = self._search_knowledge_by_vector_and_bm25(
        #         query_text=query_text,
        #         query_vector=query_vector,
        #         top_n_per_source=3,
        #         initial_fetch_limit=100,
        #         alpha_vec=0.60,
        #         beta_bm25=0.40,
        #         topic_gate=True,
        #         topic_threshold=0.30,
        #     )
        #     knowledge_results = fb
        # else:
        #     knowledge_results = self._finalize_knowledge_candidates(
        #         df=sp_df,
        #         query_text=query_text,
        #         top_n_per_source=3,
        #         final_top_n=10,
        #         #topic_gate=True,
        #         topic_gate=False,
        #         topic_threshold=0.30,
        #         source_bias=None
        #     )

        fb = self._search_knowledge_by_vector_and_bm25(
            query_text=query_text,
            query_vector=query_vector,
            top_n_per_source=3,
            initial_fetch_limit=100,
            alpha_vec=0.60,
            beta_bm25=0.40,
            #topic_gate=True,
            topic_gate=False,
            topic_threshold=0.30,
        )
        knowledge_results = fb
        # 2) BigQuery: 既存ロジックのまま
        vec_hits = self._search_similar_conversations_by_vector_bq(query_vector, top_n=VEC_TOPK)
        best_responses_df = self._rerank_with_bm25_and_blend(query_text, vec_hits, top_n=top_n)
        
        # 3) ログ
        self._log_retrieval_to_bq(
            query_text=query_text,
            knowledge_results=knowledge_results,
            best_responses=best_responses_df if isinstance(best_responses_df, pd.DataFrame) else pd.DataFrame(),
            session_id=session_id,
            message_index=message_index,
        )
        self.logger.info("検索完了")

        # --- ★ 追加・変更点: ヒット状況のメタデータを生成 ---
        
        # 公式ナレッジ(Spanner)のヒット数
        kb_hit_count = len(knowledge_results.get("ids", []))
        
        # 過去QA(BQ)のヒット数
        qa_hit_count = len(best_responses_df) if isinstance(best_responses_df, pd.DataFrame) else 0

        # 公式情報が欠落しているかどうかのフラグ
        # (QAはあるのにナレッジがない場合 True)
        is_knowledge_missing = (kb_hit_count == 0) and (qa_hit_count > 0)
        
        return {
            "knowledge": knowledge_results,
            "best_responses": best_responses_df.to_dict(orient="records")
            if isinstance(best_responses_df, pd.DataFrame) and not best_responses_df.empty else [],
            # Agent側で判断に使うためのメタデータ
            "search_meta": {
                "kb_hit_count": kb_hit_count,
                "qa_hit_count": qa_hit_count,
                "is_knowledge_missing": is_knowledge_missing  # これが重要
            }
        }


    # ---------- Logging ----------
    def _log_retrieval_to_bq(
        self,
        query_text: str,
        knowledge_results: Dict[str, Any],
        best_responses: pd.DataFrame,
        session_id: Optional[str],
        message_index: Optional[int],
    ):
        try:
            row = {
                "log_timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id,
                "message_index": message_index,
                "query_text": query_text,
                "retrieved_global_content_id": knowledge_results.get("ids", []),
                "retrieved_knowledge_url": knowledge_results.get("url", []),
                "retrieved_knowledge_bodies": knowledge_results.get("knowledges", []),
                "retrieved_knowledge_categories": knowledge_results.get("category", []),
                "retrieved_response_user_bodies": best_responses['body_user'].tolist() if not best_responses.empty else [],
                "retrieved_response_admin_bodies": best_responses['body_admin'].tolist() if not best_responses.empty else [],
                "retrieved_response_scores": best_responses['blended_score'].tolist() if not best_responses.empty else [],
            }
            errors = self.bq_client.insert_rows_json(LOG_TABLE_BQ, [row])
            if not errors:
                self.logger.info("検索ログをBigQueryに正常に記録しました。")
            else:
                self.logger.error(f"BigQueryへのログ記録中にエラー: {errors}")
        except Exception as e:
            self.logger.error(f"BigQueryログ書込エラー: {e}", exc_info=True)

    # ---------- Similar Conversations (BQ Vector) ----------
    # def _search_similar_conversations_by_vector_bq(self, query_vector: np.ndarray, top_n: int = VEC_TOPK) -> List[Dict[str, Any]]:
    #     sql = f"""
    #     SELECT conversation_id, COSINE_DISTANCE(vector, @query_vector) AS distance
    #     FROM `{VECTOR_TABLE_BQ}`
    #     ORDER BY distance
    #     LIMIT @limit
    #     """
    #     job_config = bigquery.QueryJobConfig(
    #         query_parameters=[
    #             bigquery.ArrayQueryParameter("query_vector", "FLOAT64", query_vector.tolist()),
    #             bigquery.ScalarQueryParameter("limit", "INT64", top_n),
    #         ]
    #     )
    #     try:
    #         rows = list(self.bq_client.query(sql, job_config=job_config).result())
    #         return [{"conversation_id": r.conversation_id, "distance": float(r.distance)} for r in rows]
    #     except Exception as e:
    #         self.logger.error(f"会話検索エラー: {e}", exc_info=True)
    #         return []


    # ---------- Similar Conversations (BQ Vector) 最適化入----------
    def _search_similar_conversations_by_vector_bq(self, query_vector: np.ndarray, top_n: int = VEC_TOPK) -> List[Dict[str, Any]]:
        # 検索対象期間の設定（例: 2024年以降のデータのみ対象にする）
        # ※ここを古い日付にすれば全期間検索になるが、パーティションの恩恵を受けるには日付指定が推奨
        search_since = "2024-01-01 00:00:00"
        search_end = "2025-10-01 00:00:00"

        sql = f"""
        SELECT 
            base.conversation_id, 
            distance
        FROM VECTOR_SEARCH(
            TABLE `{VECTOR_TABLE_BQ}`,
            'vector',
            (SELECT @query_vector AS vector),
            top_k => @fetch_k, 
            options => '{{"fraction_lists_to_search": 0.05}}'
        )
        WHERE 
            -- パーティションプルーニング: この日付以前のデータはスキャンされず課金されません
            base.created_at >= TIMESTAMP(@search_since, 'Asia/Tokyo')
            AND base.created_at <= TIMESTAMP(@search_end, 'Asia/Tokyo')
        ORDER BY distance ASC
        LIMIT @limit
        """
        
        # VECTOR_SEARCHは絞り込みで減る可能性があるので、要求(top_n)の倍くらい候補を取るのが定石です
        fetch_k = top_n * 2 

        # job_config = bigquery.QueryJobConfig(
        #     query_parameters=[
        #         bigquery.ArrayQueryParameter("query_vector", "FLOAT64", query_vector.tolist()),
        #         bigquery.ScalarQueryParameter("limit", "INT64", top_n),
        #         bigquery.ScalarQueryParameter("fetch_k", "INT64", fetch_k),
        #         bigquery.ScalarQueryParameter("search_since", "STRING", search_since),
        #     ]
        # )
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("query_vector", "FLOAT64", query_vector.tolist()),
                bigquery.ScalarQueryParameter("limit", "INT64", top_n),
                bigquery.ScalarQueryParameter("fetch_k", "INT64", fetch_k),
                bigquery.ScalarQueryParameter("search_since", "STRING", search_since),
                # 【修正2】search_end パラメータを追加
                bigquery.ScalarQueryParameter("search_end", "STRING", search_end),
            ]
        )
        
        try:
            # VECTOR_SEARCHの結果は base.カラム名 でアクセスします
            rows = list(self.bq_client.query(sql, job_config=job_config).result())
            return [{"conversation_id": r.conversation_id, "distance": float(r.distance)} for r in rows]
        except Exception as e:
            self.logger.error(f"会話検索エラー(Vector Search): {e}", exc_info=True)
            return []

    # ---------- Knowledge (Spanner) with Vector+BM25+Topic Gate ----------
    def _search_knowledge_by_vector_and_bm25(
        self,
        query_text: str,
        query_vector: np.ndarray,
        top_n_per_source: int = 2,
        initial_fetch_limit: int = 100,
        alpha_vec: float = 0.6,
        beta_bm25: float = 0.4,
        source_bias: Optional[Dict[str, float]] = None,
        topic_gate: bool = True,
        topic_threshold: float = 0.30,
    ) -> Dict[str, List]:
        self.logger.info(f"[kn_blend] fetch top{initial_fetch_limit} by vector...")
        sql = f"""
        SELECT
          k.global_content_id, k.title, {BODY_COLUMN_SPN} AS body, k.category, k.url, k.source_id,
          COSINE_DISTANCE({VECTOR_COLUMN_SPN}, @query_vector) AS distance,
          ARRAY(
            SELECT AS STRUCT p.property_type, p.property_value
            FROM properties AS p
            WHERE p.global_content_id = k.global_content_id
          ) AS props_struct
        FROM {KNOWLEDGES_TABLE} AS k
        WHERE {VECTOR_COLUMN_SPN} IS NOT NULL AND k.is_deprecated = False
        ORDER BY distance
        LIMIT @limit
        """
        params = {"query_vector": query_vector.tolist(), "limit": initial_fetch_limit}
        param_types_map = {
            "query_vector": param_types.Array(param_types.FLOAT64),
            "limit": param_types.INT64
        }
        
        rows: List[Dict[str, Any]] = []
        try:
            with self.spanner_db.snapshot() as snapshot:
                for r in snapshot.execute_sql(sql, params=params, param_types=param_types_map):
                    # インデックス対応:
                    # 0:id, 1:title, 2:body, 3:cat, 4:url, 5:source_id
                    # 6:distance (float)  <-- ここがズレないように維持
                    # 7:props_struct (ARRAY) <-- 最後に追加

                    raw_props = r[7] # ★最後尾
                    props_dict = {}
                    if raw_props:
                        for item in raw_props:
                            props_dict[item[0]] = item[1]

                    rows.append({
                        "id": r[0], "title": r[1], "body": r[2], "category": r[3],
                        "url": r[4], "source_id": r[5],
                        "distance": float(r[6]),  # ★ここは r[6] のまま
                        "properties": props_dict, 
                    })
        except Exception as e:
            self.logger.error(f"[kn_blend] Spanner error: {e}", exc_info=True)
            return {"url": [], "category": [], "knowledges": [], "titles": [], "ids": [], "source_ids": [], "scores": [], "properties": []}

        if not rows:
            return {"url": [], "category": [], "knowledges": [], "titles": [], "ids": [], "source_ids": [], "scores": []}

        df = pd.DataFrame(rows).drop_duplicates(subset=["id"], keep="first")
        df = df[df["body"].apply(lambda s: isinstance(s, str) and len(s.strip()) >= 20)]
        if df.empty:
            return {"url": [], "category": [], "knowledges": [], "titles": [], "ids": [], "source_ids": [], "scores": []}

        # --- BM25 ---
        q_tokens = self.tokenize_japanese_with_stopwords(self.clean_text(query_text))
        df["body_clean"] = df["body"].apply(self.clean_text)
        df["tok"] = df["body_clean"].apply(self.tokenize_japanese_with_stopwords)

        if (df["tok"].apply(len) == 0).all() or not q_tokens:
            df["bm25_raw"] = 0.0
        else:
            bm25 = BM25Okapi(df["tok"].tolist())
            df["bm25_raw"] = bm25.get_scores(q_tokens)

        # --- 正規化 + ブレンド ---
        vec_sim_raw = df["distance"].max() - df["distance"]
        df["vec_norm"] = _minmax(vec_sim_raw)
        df["bm25_norm"] = _minmax(df["bm25_raw"])

        source_bias = source_bias or {}
        df["source_bias"] = df["source_id"].map(lambda s: float(source_bias.get(str(s), 0.0)))
        df["blended_score"] = alpha_vec * df["vec_norm"] + beta_bm25 * df["bm25_norm"] + df["source_bias"]

        # --- ★ トピックゲート（静的本文判定） ---
        if topic_gate:
            primary = _primary_topic_from_query(query_text)
            if primary:
                df["topic_score"] = df.apply(lambda r: _topic_score_row(r, primary), axis=1)
                # 閾値未満は強制ドロップ（主題断定を優先）
                df = df[df["topic_score"] >= float(topic_threshold)]
                if df.empty:
                    return {"url": [], "category": [], "knowledges": [], "titles": [], "ids": [], "source_ids": [], "scores": []}
                # 主題に近いほどわずかに優遇（微差）：断定を崩さない程度に +0.05*topic_score
                df["blended_score"] = df["blended_score"] + 0.05 * df["topic_score"]

        # --- ソースのバランス抽出 → 最終降順 ---
        df = df.sort_values(["source_id", "blended_score", "title"], ascending=[True, False, True])
        balanced = df.groupby("source_id", as_index=False, group_keys=False).head(top_n_per_source)
        balanced = balanced.sort_values(["blended_score", "title"], ascending=[False, True])

        return {
            "url":        balanced["url"].tolist(),
            "category":   balanced["category"].tolist(),
            "knowledges": balanced["body"].tolist(),
            "titles":     balanced["title"].tolist(),
            "ids":        balanced["id"].tolist(),
            "source_ids": balanced["source_id"].tolist(),
            "scores":     balanced["blended_score"].tolist(),
            "properties": balanced["properties"].tolist(), # ★ 追加
        }


    def _finalize_knowledge_candidates(
        self,
        df: pd.DataFrame,
        query_text: str,
        top_n_per_source: int = 2,
        final_top_n: int = 10,
        topic_gate: bool = True,
        topic_threshold: float = 0.30,
        source_bias: Optional[Dict[str, float]] = None,
    ) -> Dict[str, List]:
        if df is None or df.empty:
            return {
                "url": [], "category": [], "knowledges": [], "titles": [],
                "ids": [], "source_ids": [], "scores": [],"properties": []
            }
        # topic gate（任意）
        if topic_gate:
            primary = _primary_topic_from_query(query_text)
            if primary:
                df["topic_score"] = df.apply(lambda r: _topic_score_row(r, primary), axis=1)
                df = df[df["topic_score"] >= float(topic_threshold)]
                if df.empty:
                    return {
                        "url": [], "category": [], "knowledges": [], "titles": [],
                        "ids": [], "source_ids": [], "scores": []
                    }
                # 主題一致に微バイアス
                df["blended_score"] = df["blended_score"] + 0.05 * df["topic_score"]
        
        # source 微バイアス（任意）
        source_bias = source_bias or {}
        df["source_bias"] = df["source_id"].map(lambda s: float(source_bias.get(str(s), 0.0)))
        df["blended_score"] = df["blended_score"] + df["source_bias"]
        # ソース分散 → 最終降順
        df = df.sort_values(["source_id", "blended_score", "title"],
                            ascending=[True, False, True])
        balanced = df.groupby("source_id",
                              as_index=False, group_keys=False).head(top_n_per_source)
        balanced = balanced.sort_values(["blended_score", "title"],
                                        ascending=[False, True]).head(final_top_n)
        return {
            "url":        balanced["url"].tolist(),
            "category":   balanced["category"].tolist(),
            "knowledges": balanced["body"].tolist(),
            "titles":     balanced["title"].tolist(),
            "ids":        balanced["id"].tolist(),
            "source_ids": balanced["source_id"].tolist(),
            "scores":     balanced["blended_score"].tolist(),
            "properties": balanced["properties"].tolist(),  # ★★★ これを追加！ ★★★
        }
        
    def _search_knowledge_fulltext_and_vector(
        self,
        query_text: str,
        query_vector: np.ndarray,
        top_n: int = 60,
        alpha_vec: float = 0.60,
        beta_ft: float = 0.40,
    ) -> pd.DataFrame:
        """
        非パーティションの Search Index(KnowledgesSearchAll) を使って
        全文 × ベクトルで候補を取得 → UNIONで結合して取りこぼしを防ぐ
        """
        q_compact = self.build_search_query(query_text, max_terms=5) or query_text
        token_expr = "TOKENLIST_CONCAT([body_token, title_token, memo_token, category_token])" if USE_TOKENLIST_CONCAT else "body_token"
        
        # SQL本体
        sql = f"""
        WITH 
        vec_cand AS (
            SELECT
                global_content_id AS id,
                source_id, content_id, title, url, category, {BODY_COLUMN_SPN} AS body,
                COSINE_DISTANCE({VECTOR_COLUMN_SPN}, @qvec) AS vdist,
                0.0 AS ft_score_raw, 
                'vec' as origin
            FROM {KNOWLEDGES_TABLE}
            WHERE is_deprecated = FALSE AND {VECTOR_COLUMN_SPN} IS NOT NULL
            ORDER BY vdist ASC
            LIMIT @limit_cand
        ),
        ft_cand AS (
            SELECT
                global_content_id AS id,
                source_id, content_id, title, url, category, {BODY_COLUMN_SPN} AS body,
                2.0 AS vdist, 
                SCORE({token_expr}, @q, enhance_query=>TRUE) AS ft_score_raw,
                'ft' as origin
            FROM {KNOWLEDGES_TABLE}@{{FORCE_INDEX={SSEARCH_INDEX_NAME}}}
            WHERE is_deprecated = FALSE 
              AND SEARCH({token_expr}, @q, enhance_query=>TRUE)
            LIMIT @limit_cand
        ),
        union_cand AS (
            SELECT * FROM vec_cand
            UNION ALL
            SELECT * FROM ft_cand
        ),
        distinct_cand AS (
            SELECT
                id,
                ANY_VALUE(source_id) as source_id,
                ANY_VALUE(content_id) as content_id,
                ANY_VALUE(title) as title,
                ANY_VALUE(url) as url,
                ANY_VALUE(category) as category,
                ANY_VALUE(body) as body,
                MIN(vdist) as vdist,
                MAX(ft_score_raw) as ft
            FROM union_cand
            GROUP BY id
        ),
        bounds AS (
            SELECT
                MIN(vdist) AS dmin, MAX(vdist) AS dmax,
                MIN(ft) AS fmin, MAX(ft) AS fmax
            FROM distinct_cand
        )
        SELECT
            c.id, c.source_id, c.content_id, c.title, c.url, c.category, c.body,
            c.vdist, c.ft,
            (@alpha * SAFE_DIVIDE(b.dmax - c.vdist, NULLIF(b.dmax - b.dmin, 0))) + 
            (@beta * SAFE_DIVIDE(c.ft - b.fmin, NULLIF(b.fmax - b.fmin, 0))) 
            AS blended_score,
            ARRAY(
                SELECT AS STRUCT p.property_type, p.property_value
                FROM properties AS p
                WHERE p.global_content_id = c.id
            ) AS props_struct
        FROM distinct_cand c
        CROSS JOIN bounds b
        ORDER BY blended_score DESC, title ASC
        LIMIT @limit_final
        """
        
        params = {
            "q": q_compact,
            "qvec": query_vector.tolist(),
            "limit_cand": int(max(top_n * 2, top_n)),
            "limit_final": int(top_n),
            "alpha": float(alpha_vec),
            "beta": float(beta_ft),
        }
        ptypes = {
            "q": param_types.STRING,
            "qvec": param_types.Array(param_types.FLOAT64),
            "limit_cand": param_types.INT64,
            "limit_final": param_types.INT64,
            "alpha": param_types.FLOAT64,
            "beta": param_types.FLOAT64,
        }
        
        rows: List[Dict[str, Any]] = []
        try:
            with self.spanner_db.snapshot() as snap:
                # ★修正箇所: query_options={"optimizer_version": "6"} を追加
                result_proxy = snap.execute_sql(
                    sql, 
                    params=params, 
                    param_types=ptypes,
                    query_options={"optimizer_version": "6"} 
                )
                
                for r in result_proxy:
                    raw_props = r[10] 
                    props_dict = {}
                    if raw_props:
                        for item in raw_props:
                            props_dict[item[0]] = item[1]
                    
                    rows.append({
                        "id": r[0], "source_id": r[1], "content_id": r[2], "title": r[3],
                        "url": r[4], "category": r[5], "body": r[6],
                        "distance": float(r[7]), 
                        "ft": float(r[8]),
                        "blended_score": float(r[9]),
                        "properties": props_dict,
                    })
                    
        except Exception as e:
            self.logger.error(f"[fulltext_vec] Spanner error: {e}", exc_info=True)
            return pd.DataFrame()
            
        if not rows:
            return pd.DataFrame()
            
        return pd.DataFrame(rows).drop_duplicates(subset=["id"], keep="first")


    # ---------- Rerank Past Answers (BQ) ----------
    def _rerank_with_bm25_and_blend(self, query_text: str, vec_hits: List[Dict[str, Any]], top_n: int = 5) -> pd.DataFrame:
        if not vec_hits:
            return pd.DataFrame()

        conv_ids = [str(h["conversation_id"]) for h in vec_hits]
        distance_map = {str(h["conversation_id"]): float(h["distance"]) for h in vec_hits}

        sql = f"""
        SELECT
          CAST(conversation_id AS STRING) AS conversation_id,
          body_user, body_admin, intercom_part_url
        FROM `{QA_TABLE_BQ}`
        WHERE CAST(conversation_id AS STRING) IN UNNEST(@conv_ids)
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("conv_ids", "STRING", conv_ids)]
        )

        try:
            df = self.bq_client.query(sql, job_config=job_config).to_dataframe()
        except Exception as e:
            self.logger.error(f"QA取得エラー: {e}", exc_info=True)
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        df["clean_user"] = df["body_user"].apply(self.clean_text)
        df["tokenized_body_user"] = df["clean_user"].apply(self.tokenize_japanese_with_stopwords)

        if (df["tokenized_body_user"].apply(len) == 0).all():
            df["clean_admin"] = df["body_admin"].apply(self.clean_text)
            df["tokenized_body_user"] = df["clean_admin"].apply(self.tokenize_japanese_with_stopwords)

        df = df[df["tokenized_body_user"].apply(len) > 0]
        if df.empty:
            return pd.DataFrame()

        query_tokens = self.tokenize_japanese_with_stopwords(self.clean_text(query_text))
        if not query_tokens:
            return pd.DataFrame()

        bm25 = BM25Okapi(df["tokenized_body_user"].tolist())
        df["bm25_raw"] = bm25.get_scores(query_tokens)

        # 正規化
        def _minmax_map(series: pd.Series) -> Dict[str, float]:
            if series.empty:
                return {}
            mn, mx = float(series.min()), float(series.max())
            if mx - mn < 1e-12:
                return {k: 0.0 for k in series.index}
            return {k: float((v - mn) / (mx - mn)) for k, v in series.items()}

        bm25_map = pd.Series(df["bm25_raw"].values, index=df["conversation_id"])
        bm25_norm = _minmax_map(bm25_map)

        max_d = max(distance_map.values()) if distance_map else 0.0
        vec_sim_raw = {cid: (max_d - distance_map.get(cid, max_d)) for cid in df["conversation_id"].tolist()}
        vec_sim_norm = _minmax_map(pd.Series(vec_sim_raw))

        df["vec_sim_norm"] = df["conversation_id"].map(lambda x: vec_sim_norm.get(x, 0.0))
        df["bm25_norm"]    = df["conversation_id"].map(lambda x: bm25_norm.get(x, 0.0))
        df["blended_score"] = BLEND_ALPHA * df["vec_sim_norm"] + BLEND_BETA * df["bm25_norm"]

        keep_k  = min(KEEP_TOPK, len(vec_hits))
        keep_ids = [str(h["conversation_id"]) for h in sorted(vec_hits, key=lambda x: x["distance"])[:keep_k]]

        df_keep = df[df["conversation_id"].isin(keep_ids)].copy()
        df_rest = df[~df["conversation_id"].isin(keep_ids)].copy().sort_values("blended_score", ascending=False)
        df_final = pd.concat([df_keep, df_rest], axis=0, ignore_index=True).head(max(1, top_n))

        return df_final[[
            "conversation_id", "body_user", "body_admin", "intercom_part_url",
            "bm25_raw", "bm25_norm", "vec_sim_norm", "blended_score"
        ]]

    # ---------- Text utils ----------
    def clean_text(self, text: str) -> str:
        if isinstance(text, str):
            text = re.sub(r"<[^>]+>", "", text)
            text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize_japanese_with_stopwords(self, text: str) -> List[str]:
        if not isinstance(text, str):
            return []
        
        # 二重防御のサニタイズ
        text = self._sanitize_text(text)
        if not text:
            return []
        tokens: List[str] = []
        
        for chunk in self._chunk_iter(text, size=3000):
            # 1回だけ自己修復リトライを許可
            for attempt in (1, 2):
                try:
                    tok = _get_tokenizer()
                    for token in tok.tokenize(chunk):
                        pos = token.part_of_speech.split(',')[0]
                        surface = token.surface.strip()
                        if not surface:
                            continue
                        if self.RE_SYMBOL.fullmatch(surface):
                            continue
                        if self.hiragana_pattern.fullmatch(surface):
                            continue
                        if self.RE_ALNUM.fullmatch(surface) and len(surface) <= 2:
                            continue
                        if len(surface) == 1:
                            continue
                        if pos in ['名詞','形容詞'] and surface not in self.stopwords:
                            tokens.append(surface)
                    break  # 正常終了したらリトライループを抜ける
                except IndexError:
                    # スレッドローカル Tokenizer を再初期化して 1 回だけ再挑戦
                    if attempt == 1:
                        self.logger.warning("[janome-indexerror] reinit thread-local tokenizer and retry once")
                        _thread_local.janome_tok = Tokenizer(mmap=True)
                        continue
                    # 2回目も失敗 → フォールバックへ
                    sig = hashlib.sha1(chunk.encode("utf-8", "ignore")).hexdigest()[:8]
                    self.logger.warning(f"[janome-indexerror] fallback regex tokenize len={len(chunk)} sig={sig}")
                    for surface in self._fallback_tokenize(chunk):
                        if not surface:
                            continue
                        if self.hiragana_pattern.fullmatch(surface):
                            continue
                        if self.RE_SYMBOL.fullmatch(surface):
                            continue
                        if self.RE_ALNUM.fullmatch(surface) and len(surface) <= 2:
                            continue
                        if len(surface) == 1:
                            continue
                        if surface not in self.stopwords:
                            tokens.append(surface)
                    break  # フォールバック後はこのチャンクを終了
        return tokens