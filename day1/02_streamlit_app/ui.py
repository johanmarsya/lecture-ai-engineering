import streamlit as st
import pandas as pd
import time
from database import save_to_db, get_chat_history, get_db_count, clear_db
from llm import generate_response
from data import create_sample_evaluation_data
from metrics import get_metrics_descriptions
from config import MODEL_NAMES

# カスタムCSS
st.markdown(
    """
    <style>
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f0f2f6;
    }
    .stApp {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .chat-card, .history-card, .feedback-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    .chat-card:hover, .history-card:hover, .feedback-card:hover {
        transform: translateY(-2px);
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: background-color 0.2s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #ced4da;
        background-color: #f8f9fa;
    }
    .stRadio > div {
        background-color: #e9ecef;
        border-radius: 8px;
        padding: 10px;
    }
    .css-1d391kg {
        background-color: #343a40;
        color: white;
        border-radius: 10px;
    }
    .css-1d391kg a {
        color: #ffffff !important;
    }
    h1, h2, h3 {
        color: #343a40;
    }
    .stSubheader {
        font-weight: 600;
        color: #495057;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# サイドバーでページナビゲーション
st.sidebar.title("ナビゲーション")
page = st.sidebar.radio(
    "ページを選択",
    ["チャット", "履歴閲覧", "データ管理"],
    label_visibility="collapsed"
)

# チャットページ
def display_chat_page(models):
    """チャットページのUIを表示"""
    st.title("💬 AIチャット")
    st.markdown("質問を入力してAIと対話しましょう！")

    # モデル選択
    selected_model_key = st.selectbox(
        "モデルを選択",
        options=list(MODEL_NAMES.keys()),
        key="model_select",
        help="使用するLLMモデルを選択してください。"
    )
    model_name = MODEL_NAMES[selected_model_key]
    st.info(f"選択中のモデル: **{selected_model_key}**")

    # モデル取得
    pipe = models.get(selected_model_key)
    if not pipe:
        st.error(f"モデル '{selected_model_key}' がロードされていません。")
        return

    # 質問入力
    with st.container():
        st.markdown("### 質問を入力")
        user_question = st.text_area(
            "あなたの質問をここに入力してください",
            key="question_input",
            height=100,
            value=st.session_state.get("current_question", "")
        )
        submit_button = st.button("🚀 質問を送信", key="submit_question")

    # セッション状態の初期化
    if "current_question" not in st.session_state:
        st.session_state.current_question = ""
    if "current_answer" not in st.session_state:
        st.session_state.current_answer = ""
    if "response_time" not in st.session_state:
        st.session_state.response_time = 0.0
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = False
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = model_name

    # 質問送信
    if submit_button and user_question:
        if not user_question.strip():
            st.warning("質問を入力してください。")
            return
        st.session_state.current_question = user_question
        st.session_state.current_answer = ""
        st.session_state.feedback_given = False
        st.session_state.selected_model = model_name

        with st.spinner("回答を生成中..."):
            try:
                answer, response_time = generate_response(pipe, user_question)
                st.session_state.current_answer = answer
                st.session_state.response_time = response_time
                st.rerun()
            except Exception as e:
                st.error(f"回答生成中にエラーが発生しました: {e}")

    # 回答表示
    if st.session_state.current_question and st.session_state.current_answer:
        with st.container():
            st.markdown("<div class='chat-card'>", unsafe_allow_html=True)
            st.markdown(f"### 回答 (モデル: {selected_model_key})")
            st.markdown(st.session_state.current_answer)
            st.info(f"応答時間: {st.session_state.response_time:.2f}秒")
            st.markdown("</div>", unsafe_allow_html=True)

            if not st.session_state.feedback_given:
                display_feedback_form()
            else:
                if st.button("次の質問へ", key="next_question"):
                    st.session_state.current_question = ""
                    st.session_state.current_answer = ""
                    st.session_state.response_time = 0.0
                    st.session_state.feedback_given = False
                    st.rerun()

def display_feedback_form():
    """フィードバック入力フォーム"""
    with st.container():
        st.markdown("<div class='feedback-card'>", unsafe_allow_html=True)
        with st.form("feedback_form"):
            st.markdown("### フィードバック")
            feedback_options = ["正確", "部分的に正確", "不正確"]
            feedback = st.radio(
                "この回答を評価してください",
                feedback_options,
                key="feedback_radio",
                horizontal=True
            )
            correct_answer = st.text_area(
                "より正確な回答（任意）",
                key="correct_answer_input",
                height=100
            )
            feedback_comment = st.text_area(
                "コメント（任意）",
                key="feedback_comment_input",
                height=100
            )
            submitted = st.form_submit_button("📤 フィードバックを送信")
            if submitted:
                is_correct = 1.0 if feedback == "正確" else (0.5 if feedback == "部分的に正確" else 0.0)
                combined_feedback = f"{feedback}"
                if feedback_comment:
                    combined_feedback += f": {feedback_comment}"

                save_to_db(
                    st.session_state.current_question,
                    st.session_state.current_answer,
                    combined_feedback,
                    correct_answer,
                    is_correct,
                    st.session_state.response_time,
                    model_name=st.session_state.selected_model
                )
                st.session_state.feedback_given = True
                st.success("フィードバックが保存されました！")
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# 履歴閲覧ページ
def display_history_page():
    """履歴閲覧ページのUI"""
    st.title("📜 チャット履歴")
    history_df = get_chat_history()

    if history_df.empty:
        st.info("まだチャット履歴がありません。")
        return

    tab1, tab2 = st.tabs(["履歴リスト", "評価分析"])
    with tab1:
        display_history_list(history_df)
    with tab2:
        display_metrics_analysis(history_df)

def display_history_list(history_df):
    """履歴リストを表示"""
    st.markdown("### 履歴リスト")
    filter_options = {
        "すべて表示": None,
        "正確なもののみ": 1.0,
        "部分的に正確なもののみ": 0.5,
        "不正確なもののみ": 0.0
    }
    display_option = st.radio(
        "表示フィルタ",
        options=filter_options.keys(),
        horizontal=True
    )

    filter_value = filter_options[display_option]
    if filter_value is not None:
        filtered_df = history_df[history_df["is_correct"].notna() & (history_df["is_correct"] == filter_value)]
    else:
        filtered_df = history_df

    if filtered_df.empty:
        st.info("選択した条件に一致する履歴はありません。")
        return

    items_per_page = 5
    total_items = len(filtered_df)
    total_pages = (total_items + items_per_page - 1) // items_per_page
    current_page = st.number_input('ページ', min_value=1, max_value=total_pages, value=1, step=1)

    start_idx = (current_page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    paginated_df = filtered_df.iloc[start_idx:end_idx]

    for i, row in paginated_df.iterrows():
        with st.expander(f"{row['timestamp']} - Q: {row['question'][:50] if row['question'] else 'N/A'}..."):
            st.markdown(f"**質問:** {row['question']}")
            st.markdown(f"**回答:** {row['answer']}")
            st.markdown(f"**モデル:** {row.get('model_name', '不明')}")
            st.markdown(f"**フィードバック:** {row['feedback']}")
            if row['correct_answer']:
                st.markdown(f"**正しい回答:** {row['correct_answer']}")

            st.markdown("---")
            cols = st.columns(3)
            cols[0].metric("正確性", f"{row['is_correct']:.1f}")
            cols[1].metric("応答時間(秒)", f"{row['response_time']:.2f}")
            cols[2].metric("単語数", f"{row['word_count']}")

            cols = st.columns(3)
            cols[0].metric("BLEU", f"{row['bleu_score']:.4f}" if pd.notna(row['bleu_score']) else "-")
            cols[1].metric("類似度", f"{row['similarity_score']:.4f}" if pd.notna(row['similarity_score']) else "-")
            cols[2].metric("関連性", f"{row['relevance_score']:.4f}" if pd.notna(row['relevance_score']) else "-")

    st.caption(f"{total_items} 件中 {start_idx+1} - {min(end_idx, total_items)} 件を表示")

def display_metrics_analysis(history_df):
    """評価指標の分析結果を表示"""
    st.markdown("### 評価指標の分析")
    analysis_df = history_df.dropna(subset=['is_correct'])
    if analysis_df.empty:
        st.warning("分析可能な評価データがありません。")
        return

    accuracy_labels = {1.0: '正確', 0.5: '部分的に正確', 0.0: '不正確'}
    analysis_df['正確性'] = analysis_df['is_correct'].map(accuracy_labels)

    st.markdown("#### 正確性の分布")
    accuracy_counts = analysis_df['正確性'].value_counts()
    if not accuracy_counts.empty:
        st.bar_chart(accuracy_counts)
    else:
        st.info("正確性データがありません。")

    st.markdown("#### モデルごとの正確性")
    if 'model_name' in analysis_df.columns:
        model_accuracy = analysis_df.groupby('model_name')['is_correct'].mean()
        st.bar_chart(model_accuracy)

    st.markdown("#### 応答時間と指標の関係")
    metric_options = ["bleu_score", "similarity_score", "relevance_score", "word_count"]
    valid_metric_options = [m for m in metric_options if m in analysis_df.columns and analysis_df[m].notna().any()]
    if valid_metric_options:
        metric_option = st.selectbox(
            "比較する指標",
            valid_metric_options,
            key="metric_select"
        )
        chart_data = analysis_df[['response_time', metric_option, '正確性']].dropna()
        if not chart_data.empty:
            st.scatter_chart(
                chart_data,
                x='response_time',
                y=metric_option,
                color='正確性',
            )
        else:
            st.info(f"選択された指標 ({metric_option}) のデータがありません。")
    else:
        st.info("比較可能な指標データがありません。")

    st.markdown("#### 評価指標の統計")
    stats_cols = ['response_time', 'bleu_score', 'similarity_score', 'word_count', 'relevance_score']
    valid_stats_cols = [c for c in stats_cols if c in analysis_df.columns and analysis_df[c].notna().any()]
    if valid_stats_cols:
        metrics_stats = analysis_df[valid_stats_cols].describe()
        st.dataframe(metrics_stats)
    else:
        st.info("統計情報を計算できるデータがありません。")

    st.markdown("#### 効率性スコア")
    if 'response_time' in analysis_df.columns and analysis_df['response_time'].notna().any():
        analysis_df['efficiency_score'] = analysis_df['is_correct'] / (analysis_df['response_time'].fillna(0) + 0.1)
        if 'id' in analysis_df.columns:
            top_efficiency = analysis_df.sort_values('efficiency_score', ascending=False).head(10)
            if not top_efficiency.empty:
                st.bar_chart(top_efficiency.set_index('id')['efficiency_score'])
            else:
                st.info("効率性スコアデータがありません。")
        else:
            st.bar_chart(analysis_df.sort_values('efficiency_score', ascending=False).head(10)['efficiency_score'])
    else:
        st.info("応答時間データがありません。")

# サンプルデータ管理ページ
def display_data_page():
    """サンプルデータ管理ページのUI"""
    st.title("🛠 データ管理")
    count = get_db_count()
    st.write(f"データベースに **{count} 件** のレコードがあります。")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 サンプルデータを追加", key="create_samples"):
            create_sample_evaluation_data()
            st.rerun()
    with col2:
        if st.button("🗑 データベースをクリア", key="clear_db_button"):
            if clear_db():
                st.rerun()

    st.markdown("### 評価指標の説明")
    metrics_info = get_metrics_descriptions()
    for metric, description in metrics_info.items():
        with st.expander(f"{metric}"):
            st.write(description)

# ページ表示
if page == "チャット":
    display_chat_page(models)
elif page == "履歴閲覧":
    display_history_page()
elif page == "データ管理":
    display_data_page()