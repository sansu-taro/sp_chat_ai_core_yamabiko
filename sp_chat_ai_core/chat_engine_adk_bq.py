# adk_chat_engine.py (修正版)
from google import genai
from google.genai import types
import uuid
import pandas as pd
import textwrap
import numpy as np
from typing import Dict, Any, Optional

from openai import AzureOpenAI

# --- 修正されたRetrieverをインポート ---
from .retriever_adk_bq import RefactoredRetriever
from . import google_secret_manager as gsm
import re

_HEADING_RE = re.compile(r'^(?P<indent>\s{0,3})(?P<hashes>#{1,6})(?=\s)', re.M)

def demote_md_headings_outside_code(text: str, offset: int = 1, min_level: int = 3) -> str:
    """
    コードフェンス外のATX見出し(#...)を offset 分だけ段下げ。
    さらに結果の見出しレベルが min_level 未満なら min_level へ引き上げる。
    ``` や ~~~ のフェンス内は変更しない。
    """
    if not text:
        return text

    out_lines = []
    in_code = False
    fence_char = None  # '```' or '~~~'

    for line in text.splitlines():
        stripped = line.lstrip()

        # フェンス開始/終了の検知
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = "```" if stripped.startswith("```") else "~~~"
            if not in_code:
                in_code, fence_char = True, marker
            elif fence_char == marker:
                in_code, fence_char = False, None
            out_lines.append(line)
            continue

        if not in_code:
            m = _HEADING_RE.match(line)
            if m:
                old = len(m.group('hashes'))
                new_level = min(max(old + offset, min_level), 6)
                new_prefix = f"{m.group('indent')}{'#' * new_level}"
                line = new_prefix + line[m.end():]

        out_lines.append(line)

    return "\n".join(out_lines)



class AdkChatbot:
    def __init__(self):
        gemini_api_key = gsm.get_secret("GEMINI_API_KEY")

        # 1) Clientを作る
        self.gclient = genai.Client(api_key=gemini_api_key)

        # 2) OpenAI(Azure)埋め込みはそのまま
        self.emb_client = AzureOpenAI(
            azure_endpoint=gsm.get_secret("AZURE_OPENAI_ENDPOINT"),
            api_key=gsm.get_secret("AZURE_OPENAI_KEY"),
            api_version="2023-05-15",
        )
        self.embedding_model_name = "text-embedding-3-large"

        self.retriever = RefactoredRetriever()
        self.sessions = {}

        # 3) system instruction と tools を config にまとめる
        system_instruction = self._build_system_instruction()

        # 自動Function Calling（Python SDK機能）を使う：

        self.chat_config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[self._get_information_for_query],  # 自動で宣言化＆実行まで面倒見てくれる
            # 必要なら tool_config でモード指定可: FunctionCallingConfig(mode="ANY" など)
        )

    def _get_embeddings_openai(self, text: str) -> np.ndarray:
        """OpenAI API を使用してテキストをベクトル化する"""
        response = self.emb_client.embeddings.create(
            input=text, model=self.embedding_model_name, encoding_format="float"
        )
        return np.array(response.data[0].embedding)

    def _build_system_instruction(self) -> str:
        """AIへのシステム指示を構築する"""
        return textwrap.dedent("""
            あなたは、SmartHRのカスタマーサポートチームに所属する、経験豊富なエキスパートです。
            あなたの行動は、以下の2つのルールによって厳格に定められています。

            ---
            ## 【最優先ルール：情報収集】
            ユーザーから質問を受けたら、他のどのタスクよりも先に、**必ず `_get_information_for_query` ツールを実行**してください。
            あなたの内部知識や推論だけで回答することは**固く禁じられています**。全ての回答は、このツールで得た「検索結果」の情報に完全に基づいている必要があります。

            ---
            ## 【第二ルール：回答の構築】
            ツールの実行後、得られた「検索結果」の情報を元に、以下の指針に従ってお客様への回答を丁寧かつプロフェッショナルに作成してください。

            1.  **結論の提示**: まず、検索結果から判断できる、質問に対する直接的な回答（機能名など）を提示します。
            2.  **機能概要の説明**: その機能が何をするためのものかを簡潔に説明します。
            3.  **詳細と注意事項**: 検索結果に手順、仕様、制約、注意点に関する記述があれば、箇条書きなどを使って分かりやすく整理して説明します。特に重要な制約（例：「〜の順番で実行してください」「〜はできません」）は、ユーザーが見落とさないように強調してください。
            4.  **先回りした質問**: 回答を提示した上で、ユーザーが次に関心を持ちそうな点や、解決に不足している可能性のある情報を予測し、「もし〜をしたい場合は、〜の情報も必要になりますが、いかがでしょうか？」といった形で、次の対話を促す質問を投げかけてください。
            5.  **常に丁寧な言葉遣い**: お客様への敬意を忘れず、常に丁寧かつ親切な言葉遣いを徹底してください。
            ---
        """)
        
    #def _get_information_for_query(self, query: str) -> tuple[str, str]:
    def _get_information_for_query(
            self,
            query: str,
            session_id: Optional[str] = None,
            message_index: Optional[int] = None,
        ) -> tuple[str, str]:
        """
        ユーザーの質問に関連する情報を検索し、AI向けと人間向けの2種類のマークダウン文字列を返します。

        Returns:
            tuple[str, str]: (AI向けのマークダウン, 人間向けのマークダウン)
        """
        print(f"ツール実行: _get_information_for_query(query='{query}')")
        
        # 1. OpenAIのモデルでクエリをベクトルに変換
        query_vector = self._get_embeddings_openai(query)
        
        # 2. テキストとベクトルをRetrieverに渡して検索
        search_results = self.retriever.search_all_sources(
            query_text=query,
            query_vector=query_vector,
            session_id=session_id,
            message_index=message_index,
        )
        
        # 3. 検索結果を整形
        
        # AI向けと人間向けのセクションをそれぞれリストに格納
        ai_readable_sections = []
        human_readable_sections = []

        # === 関連ナレッジの整形 ===
        knowledge = search_results.get("knowledge", {})
        if knowledge and knowledge.get("knowledges"):
            ai_knowledge_section = "# 関連ナレッジ\n"
            human_knowledge_section = "# 関連ナレッジ\n"
            
            knowledges = knowledge.get("knowledges", [])
            knowledge_urls = knowledge.get("url", [])
            knowledge_titles = knowledge.get("titles", [])
            knowledge_ids = knowledge.get("ids", [])
            for i, (know_text, knowledge_url, knowledge_title, global_content_id) in enumerate(
                zip(knowledges, knowledge_urls, knowledge_titles, knowledge_ids)
            ):
                # ★ ここでナレッジ本文だけ見出しを1段下げ（## の子要素になるように）
                demoted = demote_md_headings_outside_code(know_text, offset=1, min_level=3)
                # AI向け
                ai_knowledge_section += (
                    f"## ナレッジ_{i+1}\n"
                    f"**タイトル:** {knowledge_title}\n"
                    f"**本文_{i+1}:**\n{demoted}\n\n"
                )
                # 人間向け（必要ならそのまま／あるいは同じdemotedでもOK）
                human_knowledge_section += (
                    f"## ナレッジ_{i+1}\n"
                    f"**参照URL:** [ナレッジへのリンク]({knowledge_url})\n"
                    f"**content_id:** `{global_content_id}`\n"
                    f"**タイトル:** {knowledge_title}\n"
                    f"**本文_{i+1}:**\n{demoted}\n\n"
                )
            ai_readable_sections.append(ai_knowledge_section.strip())
            human_readable_sections.append(human_knowledge_section.strip())


        # 過去回答の整形 (★ここからが今回の修正箇所)
        best_responses = search_results.get("best_responses", [])
        if best_responses:
            ai_responses_section = "# 類似過去回答\n"
            human_responses_section = "# 類似過去回答\n"

            for i, resp in enumerate(best_responses):
                question = resp.get('body_user', 'N/A')
                answer = resp.get('body_admin', 'N/A')
                # intercom_part_url を取得
                url = resp.get('intercom_part_url')

                # --- AI向けの文字列を生成 ---
                ai_responses_section += f"## 過去回答_{i+1}\n"
                ai_responses_section += f"### 質問\n{question}\n\n"
                ai_responses_section += f"### 回答\n{answer}\n\n"
                
                # --- 人間向けの文字列を生成 (URLリンクを追加) ---
                human_responses_section += f"## 過去回答_{i+1}\n"
                # URLが存在する場合のみ、参照リンクを追加
                if url:
                    human_responses_section += f"**参照URL:** [該当の会話リンク]({url})\n\n"
                human_responses_section += f"### 質問\n{question}\n\n"
                human_responses_section += f"### 回答\n{answer}\n\n"
            
            # AI向けと人間向け、それぞれのリストに追加
            ai_readable_sections.append(ai_responses_section.strip())
            human_readable_sections.append(human_responses_section.strip())

        # 検索結果が何もなければ、その旨を両方の戻り値に設定
        if not ai_readable_sections:
            no_info_message = "関連する情報はナレッジベースや過去の会話ログには見つかりませんでした。"
            return no_info_message, no_info_message
        
        # AI向けと人間向けの整形済み文字列をそれぞれ結合してタプルで返す
        return "\n\n".join(ai_readable_sections), "\n\n".join(human_readable_sections)

    
    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        # GenerativeModel.start_chat ではなく Client.chats.create を使う
        self.sessions[session_id] = self.gclient.chats.create(
            model="gemini-2.5-flash",  # 例：必要なら gemini-2.5-flash 等に変更可
            config=self.chat_config,
        )
        print(f"新しいセッションを作成しました: {session_id}")
        return session_id


    def chat(self, user_input: str, session_id: str) -> Dict[str, str]:
        """
        指定されたセッションでユーザーと対話します。
        モデルの応答と、参照したコンテキストを辞書で返します。
        """
        if session_id not in self.sessions:
            raise ValueError("無効なセッションIDです。")

        chat_session = self.sessions[session_id]
        response = chat_session.send_message(user_input)

        retrieved_context = "参照された情報はありませんでした。"

        try:
            for content in reversed(chat_session.history):
                for part in content.parts:
                    if hasattr(part, 'function_response') and part.function_response.name == '_get_information_for_query':
                        response_dict = part.function_response.response
                        
                        # 辞書にキーが1つだけあると想定し、その最初の値を取得する
                        if response_dict and len(response_dict) > 0:
                            # 最初のキーを取得し、そのキーに対応する値を取得する
                            first_key = list(response_dict.keys())[0]
                            value = response_dict[first_key]
                            # 値が文字列であれば、それをコンテキストとして採用
                            if isinstance(value, str):
                                retrieved_context = value

                        # コンテキストが見つかったら、これ以上履歴を遡る必要はないのでループを抜ける
                        if retrieved_context != "参照された情報はありませんでした。":
                            break
                
                # 外側のループも抜ける
                if retrieved_context != "参照された情報はありませんでした。":
                    break

        except Exception as e:
            print(f"コンテキストの抽出中に予期せぬエラーが発生しました: {e}")
            pass

        return {
            "answer": response.text,
            "context": retrieved_context
        }
