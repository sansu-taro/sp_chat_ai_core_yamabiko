from .chat_engine_adk_bq import AdkChatbot
from google import genai
from google.genai import types


from typing import TypedDict, Annotated, List, Sequence
import operator
import json
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
            "今回のご案内でご不明点は解消されましたでしょうか？\n"
            "・同じ機能についての追加のご質問があれば、そのまま続けてお知らせください。\n"
            "・専門の担当者へのお取次ぎをご希望の場合はオペレータの連絡を希望するボタンをご選択ください。"
            # 「担当者へのエスカレーション」の文言は削除
        )
        return (answer or "").rstrip() + tail

    # ----------------------------------------------------------
    # 入口ルーター
    # ----------------------------------------------------------
    def entry_router(self, state: AgentState):
        self.logger.info("---🚪 Node: entry_router ---")
        messages = state.get("messages", [])
        has_ai_before = any(not isinstance(m, HumanMessage) for m in messages[:-1]) if messages else False
        
        if messages and isinstance(messages[-1], HumanMessage) and has_ai_before:
            phase = "after_answer"
        else:
            phase = "new"
        return {"conversation_phase": phase}
        
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
        }

    # Node: ポリシーゲート（修正：エスカレーションではなく「回答不可」として処理）
    def policy_gate(self, state: AgentState):
        self.logger.info("---🚧 Node: policy_gate (Simpified)---")
        target_q = state.get("current_query") or state["user_question"]

        prompt = f"""
        次の質問がポリシー上、AIが回答すべきでない内容（個人情報照会、契約詳細など）か判定してください。

        # 判定基準
        ## 🚫 回答不可 (need_escalation: true)
        - 特定の企業の「契約内容」や「請求金額」の確認
        - **特定の個人・従業員のデータ照会・操作依頼** (例:「Aさんの給与を教えて」「Bさんを削除して」)
        - 法律的な判断・労務コンサルティング

        ## ✅ 回答OK (need_escalation: false)
        - 一般的な機能の使い方、操作手順、仕様の質問

        質問: {target_q}
        出力形式: {{"need_escalation": true/false, "reason": "理由"}}
        """
        
        r = self._gen(prompt, response_mime_type="application/json")
        data = json.loads(getattr(r, "text", "") or "{}")
        is_ng = data.get("need_escalation", False)
        reason = data.get("reason", "")

        if is_ng:
            self.logger.info(f"    - Policy Blocked: {reason}")
            # エスカレーション機能は廃止したため、ここで「回答不可」メッセージを作って会話終了ルートへ流す
            refusal_msg = (
                "申し訳ありません。個人情報や契約詳細に関するお問い合わせ、"
                "または法的な判断を要するご質問には、AIアシスタントではお答えできません。\n"
                "お手数ですが、担当者へ直接お問い合わせいただけますようお願いいたします。"
            )
            return {
                "route_decision": "conversational", # 会話終了ルートへ
                "final_answer": refusal_msg
            }

        # OKなら元のルートを維持
        return {"route_decision": state.get("route_decision", "retrieval")}
        
    # ★ escalate_to_human メソッドは完全に削除しました ★

    # Node: 情報検索
    def retrieve(self, state: AgentState):
        self.logger.info("---🔎 Node: retrieve---")
        query = state['current_query']
        session_id = state.get('session_id')
        message_index = state.get('message_index')

        ai_context, human_context, search_meta = self.chatbot._get_information_for_query(
            query,
            session_id=session_id,
            message_index=message_index,
        )
        is_knowledge_missing = search_meta.get("is_knowledge_missing", False)
        
        existing_context = state.get('retrieved_context', '')
        updated_context = (existing_context + f"\n\n--- 検索クエリ「{query}」の結果 ---\n" + (ai_context or "")).strip()

        existing_human_context = state.get('human_readable_context', '')
        updated_human_context = (existing_human_context + "\n" + (human_context or "")).strip() if existing_human_context else (human_context or "")
        history = state.get("route_history", []) + ["retrieve"]

        return {
            "retrieved_context": updated_context,
            "human_readable_context": updated_human_context,
            "is_knowledge_missing": is_knowledge_missing,
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
                 "route_history": history
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
                 "route_history": history
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
        あなたは、SmartHRのカスタマーサポートチームに所属するエキスパートです。
        提供された「根拠情報」に基づいて、正確かつ親切な回答を生成してください。

        # 回答生成のルール
        1. **情報の統合:** - 「関連ナレッジ」や「類似過去回答」から、質問に関連する情報を探してください。
           - 完全な一致（「はい、可能です」など）がなくても、**機能の仕様や操作手順の記述から、質問に対する答えが論理的に導き出せる場合**は、それを回答として提示してください。
           - 例: 質問「管理者は編集できるか？」に対し、ナレッジに「編集画面から更新できます」とあれば、「はい、編集画面から更新可能です」と回答して構いません。

        2. **情報源の優先:**
           - 「関連ナレッジ」の情報を最優先してください。「類似過去回答」は補足として扱います。

        3. **推測の範囲:**
           - 根拠情報に全く記述がない機能や仕様については、決して創作しないでください。
           - ただし、一般的な操作（「保存ボタンを押す」など）や、文脈上明らかな主語（「操作画面」といえば通常はユーザー/管理者が操作する）については、補って説明しても構いません。

        4. **情報不足の場合:**
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
        あなたは、SmartHRの親切なカスタマーサポートアシスタントです。
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


    # Node: 評価・計画
    def grade_answer_and_plan(self, state: AgentState):
        self.logger.info("---🤔 Node: grade_answer_and_plan---")
        attempts = state.get("search_attempts", 0) + 1
        
        # 簡易実装: 回答が空ならinsufficient
        if not state.get("initial_answer"):
             return {"sufficiency_decision": "insufficient", "search_attempts": attempts}

        prompt = f"""
        あなたは、カスタマーサポートエージェントの回答をレビューする品質管理者です。
        「ユーザーの質問」に対して、「生成された回答」が解決策を提示できているかを判定してください。

        # 入力情報
        - ユーザーの質問: {state['user_question']}
        - 生成された回答: {state['initial_answer']}

        # 判定基準
        1. **Sufficient (十分)**:
           - 質問に対する具体的な手順、解決策、またはYes/Noが提示されている。
           - **【重要例外】**: 公式マニュアル(ナレッジ)がなくても、過去の問い合わせ履歴(Past QA)を引用して、具体的なエラー解決策や回避策を提示できている場合は、「十分」と判定してください。
           - 回答内で「〜でしょうか？」と状況確認の質問をしている場合も、会話を進めるために「十分」と判定してください。

        2. **Insufficient (不足)**:
           - 「情報が見つかりませんでした」「わかりません」という結論の場合。
           - **ユーザーに対して状況確認や情報の追加を求めている場合（聞き返し）。**
           - 質問と回答がかみ合っていない場合。

        # 出力 (JSON形式のみ)
        {{
            "status": "sufficient" | "insufficient",
            "next_query": "insufficientの場合のみ、次に検索すべきキーワード（※ユーザーへの質問文ではなく、検索エンジンに入力する単語）"
        }}
        """
        resp = self._gen(prompt, response_mime_type="application/json")
        result = json.loads(getattr(resp, "text", "") or "{}")
        status = result.get("status", "sufficient")
        next_q = result.get("next_query", state["current_query"])

        if status == "insufficient":
            return {
                "sufficiency_decision": "insufficient",
                "current_query": next_q,
                "search_attempts": attempts
            }
        
        return {"sufficiency_decision": "sufficient", "search_attempts": attempts}

    # Node: ファクトチェック
    def fact_check(self, state: AgentState):
        self.logger.info("---🔬 Node: fact_check---")
        # （元の実装と同じ）
        # 簡略化のため、常にOKとして通すか、元の厳密なチェックを残すかは自由ですが、
        # ここでは元のロジックを維持します。
        prompt = f"""
        あなたは、カスタマーサポートの回答を監査する、極めて厳格な品質保証（QA）の専門家です。
        あなたのタスクは2つあります。

        1.  **監査:** 「生成された回答」が「根拠情報」に基づいているか、特に「ユーザーの質問」の前提が誤っていないかを評価します。
        2.  **清書 (監査OKの場合のみ):** もし監査の結果がOK (is_grounded: true) だった場合、回答をユーザー提示用の最終形式（根拠の引用付き）に清書します。その際、必ず根拠情報に含まれるURLを使用し、回答本文中の適切な単語にハイパーリンクを適用してください。形式は必ず **Markdown** `[表示したいテキスト](URL)` としてください。（例: 「操作手順については[管理者マニュアル](https://...)を参照してください」）

        # 評価対象
        - **ユーザーの質問**: {state['user_question']}
        - **根拠情報**: {state['retrieved_context']}
        - **生成された回答 (本文のみ)**: {state['initial_answer']}

        # 監査基準 (最優先)
        - ユーザーの質問の前提（例：「Aの後にBをする」）が、根拠情報（例：「Bの後にAをする」）と矛盾している場合、回答がその矛盾を指摘せず前提を肯定していれば、**NG**です。
        - 回答に、根拠情報にない情報や拡大解釈が含まれていれば、**NG**です。

        # 出力形式 (JSON)
        監査の結果、以下のどちらかの形式で出力してください。

        ## 1. 監査がOKだった場合
        {{
            "is_grounded": true,
            "reason": "回答は根拠情報に基づいており、前提の誤りもありませんでした。",
            "formatted_answer": "（ここに、回答本文と「**根拠情報:**」の引用ブロックを含む、清書済みの最終回答を生成する）"
        }}
        
        ## 2. 監査がNGだった場合
        {{
            "is_grounded": false,
            "reason": "（ここに、NGと判断した具体的な理由を記述する。例：ユーザーの誤った前提を肯定している...）",
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

    # Node 6: rewrite_answer (修正版)
    def rewrite_answer(self, state: AgentState):
        self.logger.info("---🔧 Node: rewrite_answer (fact check)---")
        reason = (state.get('fact_check_result') or {}).get('reason', '')
        
        # ★ プロンプト修正: こちらもMarkdown形式を強制
        prompt = f"""
        ファクトチェックで指摘を受けました。指摘内容を踏まえ、必ず「根拠情報」のみで回答を**修正**し、ユーザーに提示する最終形式に清書してください。

        # リンク埋め込みのルール (最重要)
        根拠情報に含まれるURLを使用し、回答本文中の適切な単語にハイパーリンクを適用してください。
        形式は必ず **Markdown** `[表示したいテキスト](URL)` としてください。
        （例: 「操作手順については[管理者マニュアル](https://...)を参照してください」）

        # 構成指示
        1. 修正後の回答本文（リンク埋め込み済み）
        2. 回答の末尾に、根拠となった情報の要約を「**根拠情報:**」として箇条書きで記載。

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
        base = state["initial_answer"]
        final = self._append_resolution_check(base)
        return {"final_answer": final}

    # Node: 最終化 (Conversational)
    def finalize_conversational_response(self, state: AgentState):
        self.logger.info("---🏁 Node: finalize_conversational_response---")
        return {"final_answer": state['final_answer']}

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
    workflow.add_node("entry_router", agent.entry_router)
    workflow.add_node("classify_intent", agent.classify_intent)
    workflow.add_node("followup_classifier", agent.followup_classifier)
    workflow.add_node("policy_gate", agent.policy_gate)
    # workflow.add_node("escalate_to_human", agent.escalate_to_human)  <-- 削除
    
    workflow.add_node("check_ambiguity", agent.check_ambiguity) 
    workflow.add_node("generate_conversational", agent.generate_conversational_answer)
    workflow.add_node("finalize_conversational", agent.finalize_conversational_response)
    workflow.add_node("retrieve", agent.retrieve)
    workflow.add_node("generate_retrieval", agent.generate_initial_answer)
    workflow.add_node("grade_and_plan", agent.grade_answer_and_plan)
    workflow.add_node("fact_check", agent.fact_check)
    workflow.add_node("rewrite_fact", agent.rewrite_answer)
    workflow.add_node("finalize_retrieval", agent.finalize_retrieval_response)

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

    # 4. ポリシーゲート -> 検索 or 会話(拒否メッセージ)
    def pg_route(state: AgentState):
        # policy_gate内で拒否(final_answer設定済)なら会話終了へ
        if state.get("final_answer"): 
            return "finalize_conversational" # そのまま終了へ
        
        rd = state.get("route_decision", "retrieval")
        if rd == "conversational":
            return "generate_conversational"
        return "retrieve"

    workflow.add_conditional_edges("policy_gate", pg_route, {
        "finalize_conversational": "finalize_conversational",
        "generate_conversational": "generate_conversational",
        "retrieve": "retrieve"
    })

    # 5. 検索フロー
    workflow.add_edge("retrieve", "generate_retrieval")
    workflow.add_edge("generate_retrieval", "grade_and_plan")
    
    def grade_route(state: AgentState):
        if state.get("sufficiency_decision") == "insufficient":
            return "check_ambiguity"
        return "fact_check"

    workflow.add_conditional_edges("grade_and_plan", grade_route, {
        "check_ambiguity": "check_ambiguity",
        "fact_check": "fact_check"
    })

    # 6. Ambiguity -> 再検索 or 終了
    def ambiguity_route(state: AgentState):
        if state.get("route_decision") == "ambiguous":
            return "finalize_conversational" # 諦めて終了
        return "retrieve" # 再検索

    workflow.add_conditional_edges("check_ambiguity", ambiguity_route, {
        "finalize_conversational": "finalize_conversational",
        "retrieve": "retrieve"
    })

    # 7. ファクトチェック後
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