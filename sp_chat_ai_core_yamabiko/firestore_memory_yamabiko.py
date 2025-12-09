import sys
import logging
from typing import List, Dict, Any, Optional
from google.cloud import firestore
from langchain_core.messages import (
    BaseMessage, messages_to_dict, messages_from_dict, HumanMessage, AIMessage
)
# sp_chatbot.memory_base のパスは環境に合わせてください
from .chat_memory import BaseMemory 

# ログ設定
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)

# ---------------------------------------------------------
# 1. FirestoreMemory クラス
# ---------------------------------------------------------
class FirestoreMemory(BaseMemory):
    """
    セッション履歴と文脈メタデータをFirestoreで管理するクラス
    """
    def __init__(self, collection: str = "chat_sessions", project: str = None):
        # project IDが指定された場合はそれを使用、なければ環境変数から自動取得
        self.db = firestore.Client(project=project)
        self.col = self.db.collection(collection)

    # ========== BaseMemory の要件を満たすためのメソッド ==========
    def get_history(self, session_id: str) -> List[BaseMessage]:
        return self.get_session_data(session_id)["history"]

    def save_history(self, session_id: str, history: List[BaseMessage]):
        self.col.document(session_id).set(
            {"history": messages_to_dict(history)},
            merge=True
        )
    # =========================================================

    def get_session_data(self, session_id: str) -> Dict[str, Any]:
        doc = self.col.document(session_id).get()
        if not doc.exists:
            return {
                "history": [],
                "metadata": {} 
            }
        
        data = doc.to_dict()
        history_objs = messages_from_dict(data.get("history", []))
        return {
            "history": history_objs,
            "metadata": data.get("metadata", {})
        }

    def save_session_data(self, session_id: str, history: List[BaseMessage], metadata: Dict[str, Any]):
        self.col.document(session_id).set(
            {
                "history": messages_to_dict(history),
                "metadata": metadata
            },
            merge=True
        )

# ---------------------------------------------------------
# 2. 実行用関数の定義 (戻り値ありに変更)
# ---------------------------------------------------------
def run_chat_cycle(
    session_id: str, 
    user_input: str, 
    recognized_products: str,
    memory: FirestoreMemory,
    app
) -> str:  # ★戻り値をstrと定義
    
    # [A] Firestoreから前回の状態を復元
    session_data = memory.get_session_data(session_id)
    chat_history = session_data["history"]
    metadata = session_data["metadata"]

    last_answer = metadata.get("last_answer", "")
    last_context = metadata.get("last_context", "")
    last_query = metadata.get("last_query", "")

    print(f"🔙 前回のクエリ復元: {last_query}")

    # [B] ユーザー入力を履歴に追加
    chat_history.append(HumanMessage(content=user_input))

    # [C] Stateへの入力データ作成
    inputs = {
        "conversation_id": session_id,
        "recognized_products": recognized_products,
        "messages": chat_history,
        "initial_answer": last_answer,
        "retrieved_context": last_context,
        "current_query": last_query, 
        "route_history": [],
        "is_clarification_required": False
    }

    print("\n🔄 AIが回答を生成中...")
    try:
        final_state = app.invoke(inputs)
    except Exception as e:
        error_msg = f"❌ エラーが発生しました: {e}"
        print(error_msg)
        return error_msg  # エラー時も文字列を返す

    # [D] 結果の取得
    final_answer = final_state.get('final_answer', 'エラーにより回答を生成できませんでした。')
    
    # 次回のために保存すべき新しい状態
    new_last_query = final_state.get('current_query', '')
    new_last_answer = final_answer
    new_last_context = final_state.get('human_readable_context') or final_state.get('retrieved_context') or ""

    # 履歴にAIの応答を追加
    chat_history.append(AIMessage(content=final_answer))

    # [E] Firestoreに保存
    new_metadata = {
        "last_query": new_last_query,
        "last_answer": new_last_answer,
        "last_context": new_last_context
    }
    memory.save_session_data(session_id, chat_history, new_metadata)
    
    print("💾 Firestore保存完了")

    # ★ ここで最終回答を返す
    return final_answer