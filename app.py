from dotenv import load_dotenv

import openai
import streamlit as st
import uuid
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 状態（State）の定義
class State(TypedDict):
    messages: Annotated[list, add_messages]
    selected_item: str

# LLM取得（キャッシュ化）
@st.cache_resource
def get_llm():
    return ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# プロンプトテンプレートの定義
def get_prompt_template(selected_item: str):
    if selected_item == "栄養のプロ":
        system_message = """
        あなたは栄養と食事に関するプロフェッショナルです。
        ユーザーの健康のお悩みに対して、科学的根拠に基づき食事のアドバイスをしてください。
        """
    else:
        system_message = """
        あなたは運動と健康に関するプロフェッショナルです。
        ユーザーの健康のお悩みに対して、実践的なトレーニングのアドバイスをしてください。
        """
    
    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="chat_history"),
    ])

# ノード（処理）の定義
def chatbot_node(state: State):
    llm = get_llm()
    prompt = get_prompt_template(state["selected_item"])

    # 履歴を制限
    trimmer = trim_messages(
        max_tokens=1000,
        strategy="last",
        token_counter=llm,
        allow_partial=False,
        start_on="human",
    )
    trimmed_messages = trimmer.invoke(state["messages"])

    chain = prompt | llm
    response = chain.invoke({"chat_history": trimmed_messages})

    return {"messages": [response]}

# langgraph構築
@st.cache_resource
def get_graph():
    builder = StateGraph(State)

    builder.add_node("chatbot", chatbot_node)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

# langglaph実行
def invoke_graph(thread_id, user_message, selected_item):
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    input = {"messages": [user_message], "selected_item": selected_item}
    result = graph.invoke(input, config=config)

    return result["messages"][-1].content

# メイン処理
try:
    # 環境変数の読み込み
    load_dotenv()

    # -----------------------------------------
    # UIの構築
    # -----------------------------------------
    st.title("健康の相談室")

    # セッション状態の初期化
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing" not in st.session_state:
        st.session_state.processing = False

    # 画面の構築
    st.write("健康に関するお悩みはありませんか？")
    st.write("２人のプロがあなたのお悩みにお答えします。")
    st.write("")
    st.write("")
    st.write("##### 運動のプロ ")
    st.write("健康のお悩みを入力すると、あなたのお悩みにあったトレーニングのアドバイスをします。")
    st.write("##### 栄養のプロ")
    st.write("健康のお悩みを入力すると、あなたのお悩みにあった食事のアドバイスをします。")
    st.write("")

    # 過去の会話履歴の最新10件を画面に表示
    for msg in st.session_state.messages[-10:]:
        # ユーザーかAIかでアイコンや位置を変えて表示
        if isinstance(msg, HumanMessage):
            role = "user"
        else:
            role = "assistant"

        with st.chat_message(role):
            st.markdown(msg.content)

    st.divider()

    st.markdown("#### どちらのプロに相談しますか？")
    selected_item = st.radio(
        "相談するプロを選択してください",
        ["運動のプロ", "栄養のプロ"],
        disabled=st.session_state.processing 
    )

    # -----------------------------------------
    # ユーザーからの入力
    # -----------------------------------------
    input_message = st.chat_input(
        "健康のお悩みを入力してください",
        disabled=st.session_state.processing 
    )

    if input_message:
        # 入力部品をdisable表示
        st.session_state.processing = True
        user_message = HumanMessage(content=input_message)
        st.session_state.messages.append(user_message)
        st.rerun()

    if st.session_state.processing:
        st.write("")
        user_message = st.session_state.messages[-1]

        # langgraphの実行
        with st.spinner(f"{selected_item}が考え中...考え中..."):
            ai_response = invoke_graph(
                st.session_state.thread_id, 
                user_message, 
                selected_item)

        # AIメッセージをセッションに追加
        ai_message = AIMessage(content=ai_response)
        st.session_state.messages.append(ai_message)

        # 入力部品をenable表示
        st.session_state.processing = False
        st.rerun()

except openai.AuthenticationError as ae:
    st.error("【管理者向け】OpenAIのAPIキーが無効です。")
    with st.expander("詳細なエラーログ"):
        st.exception(ae)
except openai.RateLimitError as re:
    st.error("【管理者向け】OpenAIの利用制限に達しました。")
    with st.expander("詳細なエラーログ"):
        st.exception(re)
except openai.APIConnectionError as ace:
    st.error("サーバーと通信できません。ネット接続を確認してください。")
    with st.expander("詳細なエラーログ"):
        st.exception(ace)
except Exception as e:
    st.error("予期せぬエラーが発生しました。")
    with st.expander("詳細なエラーログ"):
        st.exception(e)
