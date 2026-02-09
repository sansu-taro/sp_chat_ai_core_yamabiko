import logging
from datetime import datetime, timezone,timedelta
from google.cloud import bigquery
from typing import Optional, Dict, Any

# 必要なライブラリ
# pip install google-cloud-bigquery

BIGQUERY_PROJECT_ID = "smarthr-customer-support"
BIGQUERY_DATASET_ID = "yamabiko_log"
BIGQUERY_TABLE_ID = "agent_logs"

class BigQueryLogger:
    def __init__(self,
                 project_id: str=BIGQUERY_PROJECT_ID,
                 dataset_id: str=BIGQUERY_DATASET_ID,
                 table_id: str=BIGQUERY_TABLE_ID,
                 client: Optional[bigquery.Client] = None):
        """
        Args:
            project_id: GCPプロジェクトID
            dataset_id: BigQueryデータセットID
            table_id: BigQueryテーブルID
            client: BigQueryクライアント（未指定の場合は内部で生成）
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.client = client or bigquery.Client(project=project_id)
        self.table_ref = f"{project_id}.{dataset_id}.{table_id}"
        self.logger = logging.getLogger(__name__)

    def log_interaction(self, state: Dict[str, Any]) -> None:
        """
        LangGraphの最終Stateを受け取り、BigQueryへログを送信する
        """
        try:
            # Fact Checkの結果は辞書で入っているため展開する
            # 会話ルートなどで実行されていない場合はデフォルト値を設定
            fc_result = state.get("fact_check_result") or {}
            fc_is_grounded = fc_result.get("is_grounded", None) # 実行していない場合はNone(Null)
            fc_reason = fc_result.get("reason", None)
            # ★ 追加: stateの値を安全に正規化
            route_history = state.get("route_history") or []
            if not isinstance(route_history, list):
                route_history = [str(route_history)]

            final_outcome = state.get("final_outcome")
            final_outcome_reason = state.get("final_outcome_reason")

            # --- 日時の生成 (変更点) ---
            # JST (UTC+9) のタイムゾーンを定義
            JST = timezone(timedelta(hours=9))
            # 現在のJST時刻を取得
            now_jst = datetime.now(JST)
            # DATETIME型用にフォーマット (例: "2023-10-27 15:30:00.123456")
            # ※ isoformat()だとタイムゾーン情報(+09:00)が付与され、DATETIME型への挿入でエラーになる可能性があるため明示的に除去
            created_at_str = now_jst.strftime('%Y-%m-%d %H:%M:%S.%f')

            # 行データの構築
            row = {
                "created_at": created_at_str,
                "conversation_id": state.get("conversation_id"),
                "session_id": state.get("session_id"),
                "message_index": state.get("message_index"),
                "user_question": state.get("user_question"),
                "final_answer": state.get("final_answer"),
                
                # 数値型はNoneの場合0ではなくNoneにするか、要件に合わせて調整
                "search_attempts": state.get("search_attempts", 0),
                
                "sufficiency_decision": state.get("sufficiency_decision"),
                
                # ブール値
                "is_knowledge_missing": state.get("is_knowledge_missing", False),
                "is_clarification_required": state.get("is_clarification_required", False),
                
                # ARRAY<STRING>に対応
                "route_history": state.get("route_history", []),
                
                # 監査結果
                "fact_check_result_is_grounded": fc_is_grounded,
                "fact_check_result_reason": fc_reason,

                # ★ 追加
                "final_outcome": final_outcome,
                # 任意（テーブルに追加しているなら）
                "final_outcome_reason": final_outcome_reason,
                # ★ 追加: 処理時間 (秒)
                "processing_time_sec": state.get("processing_time"),
            }

            # BigQueryへ挿入
            errors = self.client.insert_rows_json(self.table_ref, [row])
            
            if errors:
                self.logger.error(f"Failed to insert logs to BigQuery: {errors}")
            else:
                self.logger.info(f"Successfully logged conversation_id: {state.get('conversation_id')}")

        except Exception as e:
            self.logger.error(f"Exception during BigQuery logging: {e}", exc_info=True)