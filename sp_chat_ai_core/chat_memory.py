from langchain_core.messages import BaseMessage
from abc import ABC, abstractmethod

# ---------------------------------------------
# --- 会話履歴の管理（Memory Management） ---
# ---------------------------------------------
class BaseMemory(ABC):
    """
    会話履歴を管理するクラスの基本形
    """
    @abstractmethod
    def get_history(self, session_id: str) -> list[BaseMessage]:
        """指定されたセッションIDの会話履歴を取得する"""
        pass

    @abstractmethod
    def save_history(self, session_id: str, history: list[BaseMessage]):
        """指定されたセッションIDの会話履歴を保存する"""
        pass

class InMemoryMemory(BaseMemory):
    """
    【現在の実装】会話履歴をサーバーのメモリ上に保存するクラス。
    """
    def __init__(self):
        # 各セッションの会話履歴を格納する辞書
        self._conversations: dict[str, list[BaseMessage]] = {}

    def get_history(self, session_id: str) -> list[BaseMessage]:
        # 辞書から履歴を取得。なければ空のリストを返す。
        return self._conversations.get(session_id, []).copy()

    def save_history(self, session_id: str, history: list[BaseMessage]):
        # 辞書に履歴を保存する
        self._conversations[session_id] = history