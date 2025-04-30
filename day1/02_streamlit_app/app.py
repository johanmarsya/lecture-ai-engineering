# app.py
import streamlit as st
import ui                   # UIモジュール
import llm                  # LLMモジュール
import database             # データベースモジュール
import metrics              # 評価指標モジュール
import data                 # データモジュール
from config import MODEL_NAMES
from huggingface_hub import login

# --- アプリケーション設定 ---
st.set_page_config(page_title="Multi-Model Chatbot", layout="wide")

# --- Hugging Faceトークンの取得 ---
# トークンを取得してログイン
try:
    hf_token = st.secrets["huggingface"]["token"]
    login(token=hf_token)
    st.info("Hugging Faceにログインしました。")
except Exception as e:
    st.error(f"Hugging Faceのログインに失敗しました。：{e}")
    st.stop()

# --- 初期化処理 ---
# NLTKデータのダウンロード（初回起動時など）
metrics.initialize_nltk()

# データベースの初期化（テーブルが存在しない場合、作成）
database.init_db()

# データベースが空ならサンプルデータを投入
data.ensure_initial_data()

# LLMモデルのロード
models = {}
for model_key, model_name in MODEL_NAMES.items():
    models[model_key] = llm.load_model(model_name)
    if not models[model_key]:
        st.error(f"モデル '{model_name}' の読み込みに失敗しました。")
        st.stop()



# --- Streamlit アプリケーション ---
st.title("🤖 Multi-Model Chatbot with Feedback")
st.write("Gemma-2-2BまたはXGLM-564Mを使って対話できます。フィードバックも送信可能！")
st.markdown("---")

# --- サイドバー ---
st.sidebar.title("ナビゲーション")
# セッション状態を使用して選択ページを保持
if 'page' not in st.session_state:
    st.session_state.page = "チャット" # デフォルトページ

page = st.sidebar.radio(
    "ページ選択",
    ["チャット", "履歴閲覧", "サンプルデータ管理"],
    key="page_selector",
    index=["チャット", "履歴閲覧", "サンプルデータ管理"].index(st.session_state.page), # 現在のページを選択状態にする
    on_change=lambda: setattr(st.session_state, 'page', st.session_state.page_selector) # 選択変更時に状態を更新
)


# --- メインコンテンツ ---
if st.session_state.page == "チャット":
    if all(models.values()):
        ui.display_chat_page(pipe)
    else:
        st.error("チャット機能を利用できません。モデルの読み込みに失敗しました。")
elif st.session_state.page == "履歴閲覧":
    ui.display_history_page()
elif st.session_state.page == "サンプルデータ管理":
    ui.display_data_page()

# --- フッター ---
st.sidebar.markdown("---")
st.sidebar.info("開発者: Johan Marsya")