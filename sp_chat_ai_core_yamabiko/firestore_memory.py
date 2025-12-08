from typing import List, Final, Optional, Any
from google.cloud import firestore
from langchain_core.messages import (
    BaseMessage, messages_to_dict, messages_from_dict
)
from .chat_memory import BaseMemory
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime, timezone

# FIRESTORE_DATABASE_ID: Final[str] = "sp-chat-ai-memory-logs"
FIRESTORE_DATABASE_ID: Final[str] = "sp-ai-chat-logs"

class FirestoreHistoryType2Model(BaseModel):
    session_id: str
    user_id: EmailStr
    title: str
    history: List[dict] = Field(default_factory=list)
    conversation_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    class Config:
        # Pydantic V2では、未知のフィールドを無視するために `extra` を使います。
        extra = 'ignore'

class FirestoreMemory(BaseMemory):
    """
    各セッションを 1 ドキュメントに保存する簡易実装。
    ★ 長文対話では 1 MiB 上限に注意。PoC では問題なし。
    """
    def __init__(self, collection: str = "chat_sessions"):
        self.db = firestore.Client(database=FIRESTORE_DATABASE_ID) # type: ignore
        self.col = self.db.collection(collection)

    # --------------- BaseMemory ---------------
    def get_history(self, session_id: str) -> List[BaseMessage]:
        doc = self.col.document(session_id).get()
        if not doc.exists:
            return []
        data = doc.to_dict()
        # historyフィールドが存在しない場合も考慮
        return messages_from_dict(data.get("history", []))

    def save_history(self, session_id: str, history: List[BaseMessage]):
        self.col.document(session_id).set(
            {"history": messages_to_dict(history)},
            merge=True
        )

    def save_history_type2(
        self,
        session_id: str,
        history: List[BaseMessage],
        retrieved_context: str,
        session_by_user: str,
        title: str,
        iso_str: str,
        conversation_id: Optional[str] = None
    ):
        """
        セッションのメタデータと履歴を保存・更新します。
        ドキュメントが存在しない場合は新規作成し、存在する場合は履歴を上書きします。
        """
        doc_ref = self.col.document(session_id)
        doc = doc_ref.get()
        history_dicts = messages_to_dict(history)
        for msg_dict in history_dicts:
            if "created_at" not in msg_dict.get("data", {}).get("additional_kwargs", {}):
                # setdefaultを使い、ネストした辞書のキーが存在しない場合でも安全に値を追加
                msg_dict.setdefault("data", {}).setdefault("additional_kwargs", {})["created_at"] = iso_str
                msg_dict.setdefault("data", {}).setdefault("additional_kwargs", {})["retrieved_context"] = retrieved_context
        if not doc.exists:
            # ドキュメントが存在しない場合：新規作成
            data_to_create = FirestoreHistoryType2Model(
                session_id=session_id,
                user_id=session_by_user, # type: ignore
                title=title,
                created_at=iso_str,
                updated_at=iso_str,
                history=history_dicts,
                conversation_id=conversation_id,
            )
            doc_ref.set(data_to_create.model_dump())
        else:
            # ドキュメントが存在する場合：更新
            data_to_update = {
                "updated_at": iso_str,
                "history": history_dicts,
            }
            doc_ref.update(data_to_update)


    def get_history_type2(self, session_id: str) -> dict[str, Any]:
        """

        :param session_id:
        :return:
        """
        doc = self.col.document(session_id).get()
        if not doc.exists:
            return {}
        data = doc.to_dict()
        # historyフィールドが存在しない場合も考慮
        return data


    def get_histories_type2(self, user_email: str) -> List[dict[str, Any]]:
        """

        :param user_email:
        :return:
        """
        query = self.col.where("user_id", "==", user_email)
        docs = query.stream()
        return [doc.to_dict() for doc in docs]
