from .chat_engine_adk_bq import AdkChatbot
from google import genai
from google.genai import types


from typing import TypedDict, Annotated, List, Sequence
import operator
import json
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import logging

# ----------------------------------------------------------------
# 1. Stateの定義 (拡張済み)
# ----------------------------------------------------------------
class AgentState(TypedDict):
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


# ----------------------------------------------------------------
# 2. Agentクラス（新SDK対応・最小差分）
# ----------------------------------------------------------------
class SupportOperationAgent:
    def __init__(self, chatbot: AdkChatbot):
        self.chatbot = chatbot
        # AdkChatbot側で新SDKの Client を既に作っているなら流用（self.chatbot.gclient）
        # 未提供環境でも動くよう、自前でもフォールバック生成
        self.client = getattr(chatbot, "gclient", None) or genai.Client()
        self.model_name = "gemini-2.5-flash"
        self.logger = logging.getLogger(__name__)

    # --- 共通ヘルパ：新SDKでの生成呼び出しを1か所に集約 ---
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

    # Node 0: ルーター (修正済み)
    def route_query(self, state: AgentState):
        self.logger.info("---🚦 Node: route_query---")
        user_question = state['messages'][-1].content

        prompt = f"""
        ユーザーからの以下の質問が、SmartHRの製品やサービスに関する具体的な情報（操作方法、仕様、料金など）を求めるものか、
        それとも一般的な挨拶、お礼、自己紹介などの会話であるかを判断してください。
        - 具体的な情報を求めている場合: "retrieval"
        - 一般的な会話である場合: "conversational"
        ユーザーの質問: "{user_question}"
        判断結果:
        """
        response = self._gen(prompt)
        route = (getattr(response, "text", "") or "").strip().lower()

        self.logger.info(f"   - 判断: {route}")
        route_decision = "retrieval" if "retrieval" in route else "conversational"

        # 検索ループの状態を初期化して返す
        return {
            "user_question": user_question,
            "route_decision": route_decision,
            "current_query": user_question,  # 最初のクエリはユーザーの質問
            "search_attempts": 0,
            "retrieved_context": "",         # コンテキストを空で初期化
        }

    # Node 1: 情報検索 (修正済み)
    def retrieve(self, state: AgentState):
        self.logger.info("---🔎 Node: retrieve---")
        query = state['current_query']
        session_id = state.get('session_id')
        message_index = state.get('message_index')

        self.logger.info(f"   - 検索クエリ: \"{query}\"")

        ai_context, human_context = self.chatbot._get_information_for_query(
            query,
            session_id=session_id,
            message_index=message_index,
        )

        # 複数回の検索結果を追記していく
        existing_context = state.get('retrieved_context', '')
        updated_context = (existing_context + f"\n\n--- 検索クエリ「{query}」の結果 ---\n" + (ai_context or "")).strip()

        existing_human_context = state.get('human_readable_context', '')
        updated_human_context = (existing_human_context + "\n" + (human_context or "")).strip() if existing_human_context else (human_context or "")

        return {
            "retrieved_context": updated_context,
            "human_readable_context": updated_human_context,
        }

    # Node 2: 回答生成
    def generate_initial_answer(self, state: AgentState):
        self.logger.info("---✍️ Node: generate_initial_answer (with retrieval)---")
        formatted_history = "".join(
            [f"お客様: {msg.content}\n" if isinstance(msg, HumanMessage) else f"AIアシスタント: {msg.content}\n"
             for msg in state['messages']]
        )

        prompt = f"""
        あなたは、SmartHRのカスタマーサポートチームに所属する、極めて正確かつ慎重なエキスパートです。
        あなたの最優先事項は、提供された「根拠情報」に完全に基づいた、正確無比な回答を生成することです。

        # 回答生成の厳格なルール
        あなたは以下のルールを**絶対的な順序**で、一つずつ実行しなければなりません。
        1.  **トピックの一致確認:**
            - お客様の質問に含まれる主要なキーワード（製品名、機能名など）を特定します。
            - 「根拠情報」の各項目が、そのキーワードと関連しているかを確認します。
            - **お客様の質問とトピックが全く異なる根拠情報（例：質問は「採用管理」なのに、情報は「部署マスター」）は、完全に無視し、回答の根拠として絶対に使用してはなりません。**

        2.  **情報源の優先順位付け:**
            - 「根拠情報」には「関連ナレッジ」と「類似過去回答」が含まれます。
            - **必ず「関連ナレッジ」の情報を最優先**してください。「類似過去回答」はあくまで参考情報です。
            - もし情報が矛盾する場合（例：ナレッジでは「可能」、過去回答では「不可能」）、**必ず「関連ナレッジ」の内容を正として採用**してください。

        3.  **直接的な情報の探索:**
            - 優先度の高い「関連ナレッジ」の中に、質問に対して「はい、可能です」「いいえ、できません」のように直接的に回答している箇所がないかを探します。
            - もし直接的な回答が見つかった場合、それがあなたの回答の**核となる結論**です。他の情報から類推してはいけません。

        4.  **忠実な回答の生成:**
            - 上記のルールで特定した、信頼できる情報**のみ**を使用して回答を作成します。
            - **根拠情報に書かれていない事柄を推測したり、独自の解釈を加えたりすることは固く禁じられています。**
            - 特に、情報を組み合わせることで元の情報にない新たな結論（例：「新規なら可能で、過去は不可能」など）を**創作してはいけません**。

        5.  **情報不足の場合の対応:**
            - 上記の手順を踏んでも、質問に答えられる信頼できる情報が見つからない場合は、安易に回答を生成せず、「恐れ入りますが、いただいた情報からでは明確なご案内が難しい状況です。」のように、正直に回答してください。

        6.  **情報の鮮度に関する判断の禁止:**
            - あなたは、根拠情報の新旧や有効性を**自己判断してはなりません**。「この情報は古い可能性がある」といった推測は、たとえもっともらしくても固く禁じられています。
            - 情報が矛盾している場合は、その事実のみを報告し、どちらが正しいかを断定してはいけません。

        # これまでの会話履歴
        {formatted_history}
        # お客様の現在の質問
        {state['user_question']}
        # 根拠情報
        {state['retrieved_context']}

        # 上記の厳格なルールに従って生成した回答 (回答の本文のみを出力し、根拠情報自体は含めないこと):
        """
        response = self._gen(prompt, temperature=0.0)
        return {"initial_answer": getattr(response, "text", "")}

    # Node 2.5: 会話のみの回答を生成する
    def generate_conversational_answer(self, state: AgentState):
        self.logger.info("---💬 Node: generate_conversational_answer---")
        formatted_history = "".join(
            [f"お客様: {msg.content}\n" if isinstance(msg, HumanMessage) else f"AIアシスタント: {msg.content}\n"
             for msg in state['messages']]
        )

        prompt = f"""
        あなたは、SmartHRの親切なカスタマーサポートアシスタントです。
        「これまでの会話履歴」を参考に、お客様の現在の質問に対して自然な会話で応答してください。

        # これまでの会話履歴
        {formatted_history}
        # お客様の現在の質問
        {state['user_question']}
        # 応答
        """
        response = self._gen(prompt)
        return {"final_answer": getattr(response, "text", "")}

    # ▼▼▼ Node 2.8: 評価・再検索計画 ▼▼▼
    def grade_answer_and_plan(self, state: AgentState):
        self.logger.info("---🤔 Node: grade_answer_and_plan---")

        attempts = state.get('search_attempts', 0) + 1
        if attempts > 3:  # 最大試行回数
            self.logger.warning("   - 最大検索回数に達しました。ファクトチェックに進みます。")
            return {"search_attempts": attempts, "sufficiency_decision": "sufficient"}

        prompt = f"""
        あなたは、カスタマーサポートエージェントの回答をレビューするAIアシスタントです。
        「ユーザーの質問」に対して、「生成された回答」が十分に答えられているか、それとも情報不足で答えられていないかを判断してください。

        # 判断基準
        - 回答が「情報が見つかりませんでした」「わかりません」「明確なご案内が難しい」といった趣旨の内容である場合、情報は**不足**しています。
        - 回答が具体的な解決策や情報を提供している場合、情報は**十分**です。

        # 判断後のアクション
        - 情報が**十分**な場合: {{"status":"sufficient","next_query":null}}
        - 情報が**不足**している場合: {{"status":"insufficient","next_query":"不足情報を得るための新しい具体的な検索クエリ"}}

        # 入力情報
        - ユーザーの元の質問: {state['user_question']}
        - これまでに検索した情報: {state['retrieved_context']}
        - 生成された回答: {state['initial_answer']}

        # 出力 (JSON形式のみ)
        """
        resp = self._gen(prompt, response_mime_type="application/json")
        try:
            result = json.loads(getattr(resp, "text", "") or "{}")
        except Exception:
            result = {}

        self.logger.info(f"   - 評価結果: {result.get('status')}")
        if result.get('status') == 'insufficient':
            self.logger.info(f"   - 次の検索クエリ: {result.get('next_query')}")
            return {
                "sufficiency_decision": "insufficient",
                "current_query": result.get('next_query'),
                "search_attempts": attempts
            }
        else:
            return {
                "sufficiency_decision": "sufficient",
                "search_attempts": attempts
            }

    # Node 3: ファクトチェック (★ 監査と清書を統合)
    def fact_check(self, state: AgentState):
        self.logger.info("---🔬 Node: fact_check (and format)---")
        prompt = f"""
        あなたは、カスタマーサポートの回答を監査する、極めて厳格な品質保証（QA）の専門家です。
        あなたのタスクは2つあります。

        1.  **監査:** 「生成された回答」が「根拠情報」に基づいているか、特に「ユーザーの質問」の前提が誤っていないかを評価します。
        2.  **清書 (監査OKの場合のみ):** もし監査の結果がOK (is_grounded: true) だった場合、回答をユーザー提示用の最終形式（根拠の引用付き）に清書します。

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
        try:
            data = json.loads(getattr(resp, "text", "") or "{}")
        except Exception:
            data = {"is_grounded": False, "reason": "JSON parse error", "formatted_answer": None}
            
        # 監査結果をfact_check_resultに保存
        # もし監査OKなら、formatted_answer を initial_answer に上書きする
        if data.get("is_grounded") and data.get("formatted_answer"):
            self.logger.info("    - 監査OK。清書版の回答を格納します。")
            return {
                "fact_check_result": data,
                "initial_answer": data["formatted_answer"] # ★ 清書版で上書き
            }
        else:
            self.logger.info("    - 監査NG。")
            return {"fact_check_result": data}



    # Node 6: rewrite_answer
    def rewrite_answer(self, state: AgentState):
        self.logger.info("---🔧 Node: rewrite_answer (fact check)---")
        reason = (state.get('fact_check_result') or {}).get('reason', '')
        prompt = f"""
        ファクトチェックで指摘を受けました。指摘内容を踏まえ、必ず「根拠情報」のみで回答を**修正**してください。
        同時に、回答の末尾に、その回答の根拠となった「根拠情報」の主要な部分を「**根拠情報:**」として箇条書きで引用してください。
        # ユーザーの質問: {state['user_question']}
        # 根拠情報: {state['retrieved_context']}
        # 初期の回答: {state['initial_answer']}
        # 指摘内容: {reason}
        # 修正後の回答:
        """
        resp = self._gen(prompt)
        return {"initial_answer": getattr(resp, "text", "")}
        
    # ▼▼▼ 新規追加 (Node 7) ▼▼▼
    # Node 7: add_citations (根拠付与のみ)
    def add_citations_to_answer(self, state: AgentState):
        self.logger.info("---✨ Node: add_citations_to_answer (fact check OK)---")
        
        prompt = f"""
        ユーザーへの回答が生成され、ファクトチェックをパスしました。
        この回答を、ユーザーに提示する最終形式に清書してください。

        # 清書のルール
        1. 「生成された回答」の論旨はそのまま維持してください。
        2. 回答の末尾に、その回答の根拠となった「根拠情報」の主要な部分を「**根拠情報:**」として箇条書きでどの部分かわかるように引用してください。
        3. 「根拠情報」から、回答の裏付けと**直接関係のない情報**は引用しないでください。
        
        # ユーザーの質問: {state['user_question']}
        # 根拠情報: {state['retrieved_context']}
        # 生成された回答 (ファクトチェック済): {state['initial_answer']}
        
        # 清書後の最終回答 (根拠の引用を含む):
        """
        resp = self._gen(prompt)
        # 最終形式の回答を initial_answer に格納
        return {"initial_answer": getattr(resp, "text", "")}

    # Node 8: finalize_retrieval_response (★ 修正)
    def finalize_retrieval_response(self, state: AgentState):
        self.logger.info("---🏁 Node: finalize_retrieval_response---")
        # この時点での 'initial_answer' は、Node 6 または Node 7 によって
        # 既に根拠が付与された「最終回答」になっています。
        return {"final_answer": state['initial_answer']}

    # Node 9: finalize_conversational_response
    def finalize_conversational_response(self, state: AgentState):
        self.logger.info("---🏁 Node: finalize_conversational_response---")
        return {"final_answer": state['final_answer']}


# ----------------------------------------------------------------
# 3. グラフ構築 (修正済み)
# ----------------------------------------------------------------
def build_support_agent_graph(chatbot_instance: AdkChatbot):
    agent = SupportOperationAgent(chatbot_instance)
    workflow = StateGraph(AgentState)

    # ノードを追加
    workflow.add_node("route_query", agent.route_query)
    workflow.add_node("retrieve", agent.retrieve)
    workflow.add_node("generate_retrieval", agent.generate_initial_answer)
    workflow.add_node("grade_and_plan", agent.grade_answer_and_plan)
    workflow.add_node("generate_conversational", agent.generate_conversational_answer)
    workflow.add_node("fact_check", agent.fact_check)
    workflow.add_node("rewrite_fact", agent.rewrite_answer)

    # ▼▼▼ 新規ノードをグラフに追加 ▼▼▼
    #workflow.add_node("add_citations", agent.add_citations_to_answer)
    
    workflow.add_node("finalize_retrieval", agent.finalize_retrieval_response)
    workflow.add_node("finalize_conversational", agent.finalize_conversational_response)

    # エントリーポイント
    workflow.set_entry_point("route_query")

    # ルーティング
    def decide_path(state: AgentState):
        return state["route_decision"]

    workflow.add_conditional_edges(
        "route_query",
        decide_path,
        {"retrieval": "retrieve", "conversational": "generate_conversational"}
    )

    # 情報検索ルート（ループ）
    workflow.add_edge("retrieve", "generate_retrieval")
    workflow.add_edge("generate_retrieval", "grade_and_plan")

    # 評価ノードからの分岐
    def should_research_again(state: AgentState):
        if state.get("sufficiency_decision") == "insufficient":
            return "retrieve"
        else:
            return "fact_check"

    workflow.add_conditional_edges("grade_and_plan", should_research_again)
    
    # ファクトチェックからの条件分岐 (★ 修正)
    def should_rewrite_or_finalize(state: AgentState): # 関数名を変更
        if state['fact_check_result']['is_grounded']:
            # パスした場合 -> 'finalize_retrieval' へ直行
            # (fact_check ノードが清書回答を initial_answer に格納済み)
            return "finalize_retrieval" 
        else:
            # 失敗した場合 -> 既存の 'rewrite_fact' ノードへ
            return "rewrite_fact"

    workflow.add_conditional_edges("fact_check", should_rewrite_or_finalize)

    # 'rewrite_fact' は 'finalize_retrieval' に合流
    workflow.add_edge("rewrite_fact", "finalize_retrieval")
    

    # 最終ノード
    workflow.add_edge("finalize_retrieval", END)
    workflow.add_edge("generate_conversational", "finalize_conversational")
    workflow.add_edge("finalize_conversational", END)

    return workflow.compile()


def build_graph(chatbot_instance: AdkChatbot):
    return build_support_agent_graph(chatbot_instance)