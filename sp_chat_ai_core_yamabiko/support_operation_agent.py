from .chat_engine_adk_bq import AdkChatbot
from google import genai
from google.genai import types


from typing import TypedDict, Annotated, List, Sequence
import operator
import json
import time
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------------------------------------------------
# 1. Stateの定義 (エスカレーション用フィールドを削除・整理)
# ----------------------------------------------------------------
class AgentState(TypedDict):

    # ▼▼▼ APPからの受け渡しフィールド ▼▼▼
    conversation_id: str
    recognized_products: str
    
    # ▼▼▼ 管理用のフィールド ▼▼▼
    session_id: str
    message_index: int
    user_question: str
    retrieved_context: str
    human_readable_context: str
    initial_answer: str
    fact_check_result: dict
    final_answer: str
    route_decision: str
    messages: Sequence[BaseMessage]

    # ▼▼▼ 検索ループ管理用のフィールド ▼▼▼
    current_query: str          # 現在の検索クエリ
    search_attempts: int        # 検索の試行回数
    sufficiency_decision: str   # 回答が十分かどうかの判断 ("sufficient" or "insufficient")

    # ▼ 図の下側フロー対応用
    conversation_phase: str      # "new" | "after_answer"
    followup_type: str           # "resolved" | "same_feature" | "different_feature" | "escalation"
    # ▼▼▼ 追加: ナレッジ不足フラグ ▼▼▼
    is_knowledge_missing: bool      # Trueなら「ナレッジ0件＆過去ログあり」の状態

    # ★ 追加: 聞き返し発生フラグ（ログ記録用）
    is_clarification_required: bool 
    # ★ 追加: 通過したルートの履歴記録用（デバッグ・分析用）
    route_history: List[str]
    
    # ▼▼▼ Plan-and-Execute用に追加 ▼▼▼
    plan_queue: list[str]       # 未実行の検索クエリリスト
    completed_steps: list[str]  # 実行済みのステップ（ログ用）

    # ★ 追加: 最終アウトカム
    final_outcome: str            # "answered" | "clarification" | "not_found" | "refused"
    final_outcome_reason: str     # 任意（ログ分析用）
    # ★ 追加: 時間計測用
    start_time: float       # 処理開始時のUNIXタイムスタンプ
    processing_time: float  # 最終的にかかった秒数

# ----------------------------------------------------------------
# 2. Agentクラス（シンプル化）
# ----------------------------------------------------------------
class SupportOperationAgent:
    def __init__(self, chatbot: AdkChatbot):
        self.chatbot = chatbot
        self.client = getattr(chatbot, "gclient", None) or genai.Client()
        self.model_name = "gemini-2.5-flash"
        self.logger = logging.getLogger(__name__)
        
        # ★ マッピングファイル読み込みなどを削除

    # --- 共通ヘルパ ---
    def _gen(self, prompt: str, *, response_mime_type: str | None = None, temperature: float | None = 0.0):
        cfg = types.GenerateContentConfig(
            response_mime_type=response_mime_type,
            temperature=0.0 if temperature is None else temperature,
        )
        return self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=cfg,
        )

    def _append_resolution_check(self, answer: str) -> str:
        """ユーザーへの最終回答の末尾に、解決確認の一文を付与する。"""
        tail = (
            "\n\n"
            "――――\n"
            "追加のご質問の場合は、続けてこちらでお問い合わせください。\n"
            "これまでのご質問内容を踏まえてご案内いたします。（※対象機能名は再度選択いただく必要がございます）\n"
            "有人サポートに切り替える場合は「解決しましたか？」の回答後、追加で「質問する」> 「オペレーターへ接続」へお進みください。"
        )
        return (answer or "").rstrip() + tail

    # ----------------------------------------------------------
    # 入口ルーター
    # ----------------------------------------------------------
    def entry_router(self, state: AgentState):
        self.logger.info("---🚪 Node: entry_router ---")
        # 開始時刻を記録 (Stateに無ければ現在時刻)
        start_time = state.get("start_time") or time.time()
        messages = state.get("messages", [])
        has_ai_before = any(not isinstance(m, HumanMessage) for m in messages[:-1]) if messages else False
        
        if messages and isinstance(messages[-1], HumanMessage) and has_ai_before:
            phase = "after_answer"
        else:
            phase = "new"
        return {"conversation_phase": phase,
                "start_time": start_time
               }
        
    # Node: 意図分類
    def classify_intent(self, state: AgentState):
        self.logger.info("---🧭 Node: classify_intent---")
        user_question = state['messages'][-1].content
            
        prompt = f"""
        次の発話の意図を1つだけ選び、日本語のままJSONで出力してください。
        - "retrieval": 製品仕様/操作/料金などの情報照会（ナレッジ参照で回答可能）
        - "conversational": 挨拶・雑談・お礼など（検索不要）
        発話: {user_question}
        出力: {{"intent":"retrieval"|"conversational"}}
        """
        resp = self._gen(prompt, response_mime_type="application/json")
        
        try:
            data = json.loads(getattr(resp, "text", "") or "{}")
            intent = data.get("intent", "retrieval")
        except Exception:
            intent = "retrieval"
        
        route_decision = "retrieval" if intent == "retrieval" else "conversational"
        
        return {
            "user_question": user_question,
            "route_decision": route_decision,
            "current_query": user_question,
            "search_attempts": 0,
            "retrieved_context": "",
            "final_outcome": "",
            "final_outcome_reason": "",
        }


    # Node: ポリシーゲート（再々々修正版：仕様確認・ロジック照会許容）
    def policy_gate(self, state: AgentState):
        self.logger.info("---🚧 Node: policy_gate (Improved v4)---")
        target_q = state.get("current_query") or state["user_question"]

        prompt = f"""
        次の質問がポリシー上、AIが回答すべきでない内容（個人情報照会、契約詳細など）か判定してください。
        このシステムは「SmartHR」というSaaS製品の操作サポートです。

        # 判定基準
        ## 🚫 回答不可 (need_escalation: true)
        - 特定の企業の「契約内容」や「請求金額」の確認
        - **AIに対して、直接的なデータ操作や照会を依頼するもの** (例:「Aさんのデータを今すぐ直して」「私の代わりに削除して」)
          ※「AIが実行する」ことは不可能です。
        - **個別の法的・税務的な判断（コンサルティング）**

        ## ✅ 回答OK (need_escalation: false)
        - 一般的な機能の使い方、操作手順、トラブルシューティング
        - **「機能の有無」や「操作手順」に関する質問**
          - 「データを間違って消した、復旧できるか？」という質問は、AIへの作業依頼ではなく、**「復旧機能はあるか（仕様）」「どう操作すれば戻るか（手順）」を聞いている**と解釈し、OKとしてください。
          - 「削除の方法」や「訂正の方法」を聞かれた場合も、手順の案内としてOKとしてください。
        - **画面上の特定の表示やメッセージが出る「条件」「原因」「仕様」に関する質問**
        - **システム挙動の正当性確認**

        質問: {target_q}
        出力形式: {{"need_escalation": true/false, "reason": "理由"}}
        """
        
        r = self._gen(prompt, response_mime_type="application/json")
        data = json.loads(getattr(r, "text", "") or "{}")
        is_ng = data.get("need_escalation", False)
        reason = data.get("reason", "")

        if is_ng:
            self.logger.info(f"    - Policy Blocked: {reason}")
            refusal_msg = (
                "申し訳ありません。個人情報や契約詳細に関するお問い合わせ、"
                "または個別の法的・税務的な判断を要するご質問には、AIアシスタントではお答えできません。\n"
                "お手数ですが、担当者へ直接お問い合わせいただけますようお願いいたします。"
            )
            return {
                "route_decision": "conversational",
                "final_answer": refusal_msg,
                "final_outcome": "refused",
                "final_outcome_reason": reason or "policy_gate_blocked",
            }

        return {"route_decision": state.get("route_decision", "retrieval")}

    # ----------------------------------------------------------
    # ★ NEW: Planning Node (検索前のタスク分解)
    # ----------------------------------------------------------
    def make_plan(self, state: AgentState):
        self.logger.info("---📅 Node: make_plan ---")
        q = state["user_question"]
        
        # 複合的な質問かどうかを判断し、検索クエリのリストを作成する
        # ※ ここで「検索不要」と判断されれば空リストを返して会話へ直行させる制御も可能
        prompt = f"""
        あなたは検索プランナーです。ユーザーの質問に回答するために必要な「検索クエリ」を洗い出し、実行順にリスト化してください。
        
        # 方針
        - 単純な質問であれば、クエリは1つで十分です。
        - 複合的な質問（例：「Aの設定方法とBの削除方法」）の場合、それぞれのクエリに分解してください。
        - 質問が抽象的な場合、具体的ないくつかのキーワードに分解しても構いません。
        - **最大5ステップ**まで設定可能です。網羅性を重視してください。
        
        # ユーザーの質問
        {q}
        
        # 出力形式 (JSON)
        {{
            "queries": ["クエリ1", "クエリ2", ...]
        }}
        """
        
        resp = self._gen(prompt, response_mime_type="application/json", temperature=0.0)
        try:
            data = json.loads(getattr(resp, "text", "") or "{}")
            queries = data.get("queries", [])
        except:
            # エラー時は元の質問をそのまま1つのクエリとして扱う
            queries = [q]
            
        # 空の場合は最低1つ入れる（検索ルートに来ている前提のため）
        if not queries:
            queries = [q]

        queries = queries[:5]
        self.logger.info(f"    - Plan created: {queries}")
        
        return {
            "plan_queue": queries,
            "completed_steps": []
        }

    # ----------------------------------------------------------
    # ★ NEW: Execution Dispatcher (計画実行ルーター)
    # ----------------------------------------------------------
    def execute_dispatch(self, state: AgentState):
        """
        plan_queue から次のクエリを取り出し、retrieve へ渡す準備をする。
        キューが空なら generate へ進むためのフラグを返す。
        """
        queue = state.get("plan_queue", [])
        
        if not queue:
            # 計画完了 -> 回答生成へ
            return {"route_decision": "ready_to_answer"}
        
        # 次のタスクを取り出す
        next_query = queue[0]
        remaining_queue = queue[1:]
        
        self.logger.info(f"---🚀 Dispatch: Next query -> '{next_query}' ---")
        
        return {
            "current_query": next_query,   # これが retrieve ノードで使われる
            "plan_queue": remaining_queue, # キューを更新
            "route_decision": "continue_search"
        }
        

    # ----------------------------------------------------------
    # Node: 情報検索 (改修: Append & Flag Merge)
    # ----------------------------------------------------------
    def retrieve(self, state: AgentState):
        self.logger.info("---🔎 Node: retrieve (Plan Execution)---")
        query = state['current_query']
        conversation_id = state.get('conversation_id')
        session_id =  state.get('session_id')
        message_index = state.get('message_index')

        # 1. 検索実行
        ai_context, human_context, search_meta = self.chatbot._get_information_for_query(
            query,
            conversation_id=conversation_id,
            session_id=session_id,
            message_index=message_index,
        )

        # 2. フラグのOR統合 (過去にTrueならずっとTrue)
        previous_missing = state.get("is_knowledge_missing", False)
        current_missing = search_meta.get("is_knowledge_missing", False)
        integrated_missing_flag = previous_missing or current_missing
        
        if current_missing:
            self.logger.warning(f"    ⚠️ Query '{query}' hit NO official knowledge.")

        # 3. コンテキストの追記 (Append)
        existing_context = state.get('retrieved_context', '')
        # 明示的に区切り線を入れる
        updated_context = (existing_context + f"\n\n--- 検索クエリ「{query}」の結果 ---\n" + (ai_context or "")).strip()

        existing_human_context = state.get('human_readable_context', '')
        updated_human_context = (existing_human_context + "\n" + (human_context or "")).strip() if existing_human_context else (human_context or "")
        
        history = state.get("route_history", []) + ["retrieve"]

        return {
            "retrieved_context": updated_context,
            "human_readable_context": updated_human_context,
            "is_knowledge_missing": integrated_missing_flag,
            "route_history": history,
        }

    # Node: 曖昧性チェック (★ここを修正)
    def check_ambiguity(self, state: AgentState):
        self.logger.info("---⚖️ Node: check_ambiguity---")
        history = state.get("route_history", []) + ["check_ambiguity"]

        attempts = state.get("search_attempts", 0)
        max_attempts = 3
        is_last_try = (attempts >= max_attempts)

        # 直前のAI回答（GradeでInsufficientとされたもの）
        initial_answer = state.get("initial_answer", "")

        # ★ プロンプト修正:
        # 単なる情報不足(Clear)なのか、文脈的に曖昧で聞き返す必要がある(Ambiguous)のかを厳密に判定
        # 過去ログ由来の「不適切な聞き返し」を除外するため、検索結果(retrieved_context)に照らして妥当性もチェック
        prompt = f"""
        現在の回答候補は「不十分」と判定されました。
        これが「検索不足」によるものか、質問が「曖昧」で絞り込めないためか判定してください。

        # 判断材料
        - ユーザーの質問: {state['user_question']}
        - 現在の検索結果: {state['retrieved_context'][:5000]}
        - 生成された回答候補: {initial_answer}

        # アクションの決定
        1. **Ambiguous (聞き返しが必要)**: 
           - 検索結果に「Aの場合はX、Bの場合はY」といった分岐情報があり、ユーザーの現状が不明なため回答を一意に絞れない場合。
           - **注意**: 生成された回答候補が「聞き返し」を行っていても、それが検索結果に基づかない不適切なもの（他製品の仕様など）である場合は、ここを選ばずに "Clear" (再検索) を選んでください。
           
        2. **Clear (再検索が必要)**: 
           - 単に情報が見つかっていない場合。
           - 生成された回答候補の「聞き返し」が的外れな場合。

        # 出力 (JSON)
        {{
            "status": "ambiguous" | "clear",
            "clarification_question": "ambiguousの場合のみ、ユーザーへの丁寧な聞き返し文を作成してください。"
        }}
        """
        
        resp = self._gen(prompt, response_mime_type="application/json")
        data = json.loads(getattr(resp, "text", "") or "{}")
        status = data.get("status", "clear")

        if status == "ambiguous":
             self.logger.info("    - 判定: Ambiguous -> ユーザーへ聞き返しを実行")
             clarification_msg = data.get("clarification_question", "詳細をお聞かせいただけますか？")
             
             return {
                 "route_decision": "ambiguous",
                 "final_answer": clarification_msg,
                 "is_clarification_required": True, # ★ フラグを立てる
                 "route_history": history,
                 "final_outcome": "clarification",
                 "final_outcome_reason": "need_more_context",
             }
        
        # Ambiguousでない場合
        if is_last_try:
             # もう検索回数上限なら諦める
             self.logger.info("    - 判定: Clearだが回数切れ -> 終了")
             fallback_msg = (
                 "申し訳ありません。何度か検索を試みましたが、"
                 "ご質問に対する明確な情報を見つけることができませんでした。"
             )
             return {
                 "route_decision": "ambiguous", # 強制終了ルートへ
                 "final_answer": fallback_msg,
                 "is_clarification_required": False,
                 "route_history": history,
                 "final_outcome": "not_found",
                 "final_outcome_reason": "max_attempts_reached",
             }
        
        # まだ検索できるなら再検索
        self.logger.info("    - 判定: Clear -> 再検索ループ")
        return {
            "route_decision": "clear",
            "is_clarification_required": False,
            "route_history": history
        }

    # Node: 回答生成
    # Node 2: 回答生成
    def generate_initial_answer(self, state: AgentState):
        self.logger.info("---✍️ Node: generate_initial_answer (with retrieval)---")
        formatted_history = "".join(
            [f"お客様: {msg.content}\n" if isinstance(msg, HumanMessage) else f"AIアシスタント: {msg.content}\n"
             for msg in state['messages']]
        )

        prompt = f"""
        あなたは、SmartHRのカスタマーサポートチームに所属するエキスパートの「AI」さんです。
        提供された「根拠情報」に基づいて、正確かつ親切な回答を生成してください。

        # 回答生成のルール
        1. **システム確認・個別調査の禁止と誘導:**
           - 根拠情報の中に「システム側で確認します」「サブドメインや組織図名を教えてください」といった**個別の調査や確認を申し出る記述**がある場合、**AIであるあなたはそれを絶対に再現してはいけません。**
           - 代わりに、**「この件はシステム的な確認が必要となる可能性があるため、有人サポート（オペレーター）へ直接お問い合わせください」** という案内へ書き換えてください。
           - 決して「私（AI）が確認しますので情報を教えてください」と言わないでください。
        2. **情報の統合:** - 「関連ナレッジ」や「類似過去回答」から、質問に関連する情報を探してください。
           - 完全な一致（「はい、可能です」など）がなくても、**機能の仕様や操作手順の記述から、質問に対する答えが論理的に導き出せる場合**は、それを回答として提示してください。
           - 例: 質問「管理者は編集できるか？」に対し、ナレッジに「編集画面から更新できます」とあれば、「はい、編集画面から更新可能です」と回答して構いません。

        3. **情報源の優先:**
           - 「関連ナレッジ」の情報を最優先してください。「類似過去回答」は補足として扱います。

        4. **推測の範囲:**
           - 根拠情報に全く記述がない機能や仕様については、決して創作しないでください。
           - ただし、一般的な操作（「保存ボタンを押す」など）や、文脈上明らかな主語（「操作画面」といえば通常はユーザー/管理者が操作する）については、補って説明しても構いません。

        5. **情報不足の場合:**
           - 上記を踏まえても答えが見つからない場合のみ、「恐れ入りますが、いただいた情報からでは明確なご案内が難しい状況です。」と回答してください。

        # これまでの会話履歴
        {formatted_history}
        # お客様の現在の質問
        {state['user_question']}
        # 根拠情報
        {state['retrieved_context']}

        # 回答:
        """
        response = self._gen(prompt, temperature=0.0)
        return {"initial_answer": getattr(response, "text", "")}

    # Node: 会話のみの回答
    def generate_conversational_answer(self, state: AgentState):
        # もしpolicy_gateで既にfinal_answerが設定されていたら生成をスキップ
        if state.get("final_answer"):
            return {}

        self.logger.info("---💬 Node: generate_conversational_answer---")
        
        formatted_history = "".join(
            [f"お客様: {msg.content}\n" if isinstance(msg, HumanMessage) else f"AIアシスタント: {msg.content}\n"
             for msg in state['messages']]
        )

        prompt = f"""
        あなたは、SmartHRの親切なカスタマーサポートアシスタント「AI」さんです。
        「これまでの会話履歴」を参考に、お客様の現在の質問に対して自然な会話で応答してください。

        # 注意事項
        - もしお客様が「担当者につないでほしい」「電話したい」等の要望を出している場合は、
          「申し訳ありませんが、現在はAIによる自動応答のみとなっております。このチャットで解決できることがあればお教えください」
          といった趣旨で、丁寧にお断りしてください。

        # これまでの会話履歴
        {formatted_history}
        # お客様の現在の質問
        {state['user_question']}
        # 応答
        """
        response = self._gen(prompt)
        return {"final_answer": getattr(response, "text", "")}


    # Node: 評価・計画 (修正版)
    def grade_answer_and_plan(self, state: AgentState):
        self.logger.info("---🤔 Node: grade_answer_and_plan---")
        attempts = state.get("search_attempts", 0) + 1
        
        # 簡易実装: 回答が空ならinsufficient
        if not state.get("initial_answer"):
             # ★修正: ここでも再検索できるようにキューに入れる（元の質問など）
             return {
                 "sufficiency_decision": "insufficient", 
                 "search_attempts": attempts,
                 "plan_queue": [state.get("user_question")] 
             }

        prompt = f"""
        あなたは、カスタマーサポートエージェントの回答をレビューする品質管理者です。
        「ユーザーの質問」に対して、「生成された回答」が解決策を提示できているかを判定してください。

        # 入力情報
        - ユーザーの質問: {state['user_question']}
        - 生成された回答: {state['initial_answer']}

        # 判定基準
        1. **Sufficient (十分)**:
           - 質問に対する具体的な手順、解決策、またはYes/Noが提示されている。
           - 公式マニュアル(ナレッジ)がなくても、過去の問い合わせ履歴(Past QA)を引用して回答できている場合は「十分」とする。
           - 文脈上、ユーザーへの聞き返しが必要で、適切な質問をしている場合も「十分」とする。

        2. **Insufficient (不足)**:
           - 「情報が見つかりませんでした」「わかりません」という結論の場合。
           - 質問と回答がかみ合っていない場合。

        # 出力 (JSON形式のみ)
        {{
            "status": "sufficient" | "insufficient",
            "next_query": "insufficientの場合のみ、次に検索すべきキーワード"
        }}
        """
        resp = self._gen(prompt, response_mime_type="application/json")
        result = json.loads(getattr(resp, "text", "") or "{}")
        status = result.get("status", "sufficient")
        next_q = result.get("next_query", state["user_question"]) # fallbackは元の質問

        if status == "insufficient":
            self.logger.info(f"    - 判定: Insufficient -> New Plan: {next_q}")
            return {
                "sufficiency_decision": "insufficient",
                "current_query": next_q,
                "plan_queue": [next_q], # ★重要: ここでキューに追加することで、Dispatchがretrieveへ誘導する
                "search_attempts": attempts
            }
        
        return {"sufficiency_decision": "sufficient", "search_attempts": attempts}

    
    # ----------------------------------------------------------
    # Node: ファクトチェック (URLフィルタリング追加版)
    # ----------------------------------------------------------
    def fact_check(self, state: AgentState):
        self.logger.info("---🔬 Node: fact_check---")
        
        prompt = f"""
        あなたは、カスタマーサポートの回答を監査する、極めて厳格な品質保証（QA）の専門家「AI」さんです。
        あなたのタスクは2つあります。

        1.  **監査:** 「生成された回答」が「根拠情報」に基づいているか、特に「ユーザーの質問」の前提が誤っていないかを評価します。
        2.  **清書 (監査OKの場合のみ):** もし監査の結果がOK (is_grounded: true) だった場合、回答をユーザー提示用の最終形式に清書します。

        # 評価対象
        - **ユーザーの質問**: {state['user_question']}
        - **根拠情報**: {state['retrieved_context']}
        - **生成された回答 (本文のみ)**: {state['initial_answer']}

        # 監査基準 (最優先)
        - ユーザーの質問の前提（例：「Aの後にBをする」）が、根拠情報（例：「Bの後にAをする」）と矛盾している場合、回答がその矛盾を指摘せず前提を肯定していれば、**NG**です。
        - 回答に、根拠情報にない情報や拡大解釈が含まれていれば、**NG**です。
        - 回答内で「**弊社システム側で確認します**」「**サブドメインを教えてください**」「**調査します**」といった、AIには実行不可能なシステム調査や個人情報の聴取を行おうとしていないか？もし含まれていれば **NG** です。「システム確認が必要な場合は有人サポートへ誘導するべき」と指摘してください。
        
        # 【重要】URL引用とリンクのルール (★ここを厳守)
        - 回答内にリンクを含める際は、Markdown形式（[]()）は使用しないでください。
        - **「記事タイトル (URL)」** の形式で記述してください。
        - 回答内にリンクを埋め込む際は、必ず **`https://support.smarthr.jp/ja/help/articles` で始まる公式ヘルプページのURLのみ** を使用してください。
        - `https://app.intercom.com` やその他のURL及びタイトルは、社内用または顧客閲覧不可のため、**絶対に引用・リンクしないでください**。
        - 根拠が「過去の回答(Past QA)」しかない場合は、内容は参考にして回答を作成し、**リンクは貼らないでください**。
        - 文字の強調（太字 **text** 等）や見出し（#）などのMarkdown記法は一切使用せず、**プレーンテキスト**で出力してください。
        
        # 出力形式 (JSON)
        監査の結果、以下のどちらかの形式で出力してください。

        ## 1. 監査がOKだった場合
        {{
            "is_grounded": true,
            "reason": "回答は根拠情報に基づいており、URLの規定も守られています。",
            "formatted_answer": "（ここに、ルールに則って清書済みの最終回答を生成する）"
        }}
        
        ## 2. 監査がNGだった場合
        {{
            "is_grounded": false,
            "reason": "（ここに、NGと判断した具体的な理由を記述する。例：社内用URLが含まれている、根拠と矛盾する...）",
            "formatted_answer": null
        }}

        # あなたの出力 (JSON形式のみ):
        """
        resp = self._gen(prompt, response_mime_type="application/json")
        data = json.loads(getattr(resp, "text", "") or "{}")
        
        if data.get("is_grounded"):
            return {"fact_check_result": data, "initial_answer": data.get("formatted_answer", "")}
        else:
            return {"fact_check_result": data}

    # ----------------------------------------------------------
    # Node: rewrite_answer (URLフィルタリング追加版)
    # ----------------------------------------------------------
    def rewrite_answer(self, state: AgentState):
        self.logger.info("---🔧 Node: rewrite_answer (fact check)---")
        reason = (state.get('fact_check_result') or {}).get('reason', '')
        
        prompt = f"""
        ファクトチェックで指摘を受けました。指摘内容を踏まえ、必ず「根拠情報」のみで回答を**修正**し、ユーザーに提示する最終形式に清書してください。

        # 【重要】形式のルール (★ここを厳守)
        1. **Markdown禁止:** 太字、斜体、見出しなどのMarkdown記法は使用しないでください。
        2. **URLの記述:** `[テキスト](URL)` ではなく、**`テキスト (URL)`** の形式で記述してください。
        3. **許可されるURL:** `https://support.smarthr.jp/ja/help/articles` で始まるURLのみ使用可能です。

        # 構成指示
        1. 修正後の回答本文（公式ヘルプへのリンクのみ埋め込み可）

        # ユーザーの質問: {state['user_question']}
        # 根拠情報: {state['retrieved_context']}
        # 初期の回答: {state['initial_answer']}
        # 指摘内容: {reason}
        
        # 修正後の回答:
        """
        resp = self._gen(prompt)
        return {"initial_answer": getattr(resp, "text", "")}

    # Node: 最終化 (Retrieval)
    def finalize_retrieval_response(self, state: AgentState):
        self.logger.info("---🏁 Node: finalize_retrieval_response---")
        # ★ 時間計算
        start_ts = state.get("start_time", time.time())
        duration = time.time() - start_ts
        
        base = state["initial_answer"]
        final = self._append_resolution_check(base)
        return {"final_answer": final,
                "final_outcome": state.get("final_outcome") or "answered",
                "final_outcome_reason": state.get("final_outcome_reason") or "",
                "processing_time": duration,  # ★ 計算結果をStateへ
               }

    # Node: 最終化 (Conversational)
    def finalize_conversational_response(self, state: AgentState):
        self.logger.info("---🏁 Node: finalize_conversational_response---")
        # ★ 時間計算
        start_ts = state.get("start_time", time.time())
        duration = time.time() - start_ts
        
        return {"final_answer": state['final_answer'],
                "final_outcome": state.get("final_outcome") or "answered",
                "final_outcome_reason": state.get("final_outcome_reason") or "",
                "processing_time": duration,
               }

    # Node: フォローアップ分類
    def followup_classifier(self, state: AgentState):
        self.logger.info("---🔁 Node: followup_classifier ---")
        last_user = state["messages"][-1].content
        
        # 直前のクエリを取得（文脈維持のため）
        previous_query = state.get("current_query", "")

        prompt = f"""
        次のお客様の発話タイプを選んでください。
        - "resolved": 解決報告・お礼
        - "followup": 追加質問（同じ話題の深掘り）
        - "new_topic": 全く別の話題への転換
        - "escalation": 担当者へ繋いでほしいという要望
        発話: {last_user}
        出力: {{"decision": "resolved"|"followup"|"new_topic"|"escalation"}}
        """
        r = self._gen(prompt, response_mime_type="application/json")
        data = json.loads(getattr(r, "text", "") or "{}")
        decision = data.get("decision", "followup")

        # デフォルトは検索クエリをそのまま使う
        next_query = last_user

        if decision == "resolved":
            route = "conversational"
        elif decision == "escalation":
            # エスカレーション希望時は会話ルートへ（AIが断りを入れる）
            route = "conversational"
        elif decision == "followup":
            # ★重要修正: 同じ話題の深掘りなら、前回のクエリと結合して検索精度を保つ
            # 例: "従業員登録" + "一括でできる？"
            route = "retrieval"
            if previous_query:
                next_query = f"{previous_query} {last_user}"
        else: # new_topic
            # 話題転換なら、新しい発言だけで検索する
            route = "retrieval"
            next_query = last_user
            
        return {
            "route_decision": route,
            "user_question": last_user,
            "current_query": next_query # 文脈考慮済みのクエリ
        }


# ----------------------------------------------------------------
# 3. グラフ構築 (escalate_to_human ノード削除版)
# ----------------------------------------------------------------

def build_support_agent_graph(chatbot_instance: AdkChatbot):
    agent = SupportOperationAgent(chatbot_instance)
    workflow = StateGraph(AgentState)

    # === ノード登録 ===
    # 既存ノード
    workflow.add_node("entry_router", agent.entry_router)
    workflow.add_node("classify_intent", agent.classify_intent)
    workflow.add_node("followup_classifier", agent.followup_classifier)
    workflow.add_node("policy_gate", agent.policy_gate)
    
    # ★追加: Plan & Execute ノード
    workflow.add_node("make_plan", agent.make_plan)
    workflow.add_node("execute_dispatch", agent.execute_dispatch)

    # 検索・生成系ノード
    workflow.add_node("retrieve", agent.retrieve)
    workflow.add_node("generate_retrieval", agent.generate_initial_answer)
    workflow.add_node("grade_and_plan", agent.grade_answer_and_plan)
    workflow.add_node("check_ambiguity", agent.check_ambiguity) 
    workflow.add_node("fact_check", agent.fact_check)
    workflow.add_node("rewrite_fact", agent.rewrite_answer)
    workflow.add_node("finalize_retrieval", agent.finalize_retrieval_response)
    
    workflow.add_node("generate_conversational", agent.generate_conversational_answer)
    workflow.add_node("finalize_conversational", agent.finalize_conversational_response)

    # === エントリーポイント ===
    workflow.set_entry_point("entry_router")

    # 1. 入口
    def entry_route(state: AgentState):
        return "after_answer" if state.get("conversation_phase") == "after_answer" else "new"

    workflow.add_conditional_edges("entry_router", entry_route, {
        "new": "classify_intent",
        "after_answer": "followup_classifier",
    })

    # 2. 意図分類 -> ポリシー
    workflow.add_edge("classify_intent", "policy_gate")

    # 3. フォローアップ -> ポリシー or 会話
    def followup_route(state: AgentState):
        if state.get("route_decision") == "conversational":
            return "generate_conversational"
        return "policy_gate"

    workflow.add_conditional_edges("followup_classifier", followup_route, {
        "generate_conversational": "generate_conversational",
        "policy_gate": "policy_gate",
    })

    # 4. ポリシーゲート -> 【修正】検索が必要なら「make_plan」へ
    def pg_route(state: AgentState):
        if state.get("final_answer"): 
            return "finalize_conversational"
        
        rd = state.get("route_decision", "retrieval")
        if rd == "conversational":
            return "generate_conversational"
        
        # ★ここが重要: いきなり retrieve せず、まずは計画(Plan)へ
        return "make_plan"

    workflow.add_conditional_edges("policy_gate", pg_route, {
        "finalize_conversational": "finalize_conversational",
        "generate_conversational": "generate_conversational",
        "make_plan": "make_plan" 
    })

    # 5. Planning -> Dispatch (計画したら実行管理へ)
    workflow.add_edge("make_plan", "execute_dispatch")

    # 6. Dispatch ループ (実行管理による振り分け)
    def dispatch_route(state: AgentState):
        # まだキューにクエリが残っていれば検索へ、なければ回答生成へ
        if state.get("route_decision") == "continue_search":
            return "retrieve"
        return "generate_retrieval"

    workflow.add_conditional_edges("execute_dispatch", dispatch_route, {
        "retrieve": "retrieve",
        "generate_retrieval": "generate_retrieval"
    })

    # ★重要: 検索が終わったら Dispatch に戻る (次のクエリがあるか確認するため)
    workflow.add_edge("retrieve", "execute_dispatch")

    # 7. 回答生成 -> 評価(Grade)
    workflow.add_edge("generate_retrieval", "grade_and_plan")
    
    # 8. 評価結果による分岐 (ここでの再検索は、Plan外の補正なので retrieve へ直接戻しても良いが、
    #    今回は grade_and_plan で plan_queue に追加する実装にしたため dispatch へ戻す)
    def grade_route(state: AgentState):
        if state.get("sufficiency_decision") == "insufficient":
            return "check_ambiguity"
        return "fact_check"

    workflow.add_conditional_edges("grade_and_plan", grade_route, {
        "check_ambiguity": "check_ambiguity",
        "fact_check": "fact_check"
    })

    # 9. Ambiguity -> 再検索(Dispatch経由) or 終了
    def ambiguity_route(state: AgentState):
        if state.get("route_decision") == "ambiguous":
            return "finalize_conversational" 
        
        # クリア(再検索)の場合は、Dispatchへ戻して再検索を実行
        return "execute_dispatch"

    workflow.add_conditional_edges("check_ambiguity", ambiguity_route, {
        "finalize_conversational": "finalize_conversational",
        "execute_dispatch": "execute_dispatch"
    })

    # 10. ファクトチェック後
    def fact_route(state: AgentState):
        if state['fact_check_result'].get('is_grounded'):
            return "finalize_retrieval"
        return "rewrite_fact"

    workflow.add_conditional_edges("fact_check", fact_route, {
        "finalize_retrieval": "finalize_retrieval",
        "rewrite_fact": "rewrite_fact"
    })

    workflow.add_edge("rewrite_fact", "finalize_retrieval")
    workflow.add_edge("finalize_retrieval", END)
    
    workflow.add_edge("generate_conversational", "finalize_conversational")
    workflow.add_edge("finalize_conversational", END)

    return workflow.compile()


# ファイルの末尾に追加
def build_graph(chatbot_instance: AdkChatbot):
    return build_support_agent_graph(chatbot_instance)