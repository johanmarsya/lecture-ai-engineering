import torch
from transformers import pipeline
import streamlit as st
import time
import logging
from config import MODEL_NAMES

# ロギング設定
logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

@st.cache_resource
def load_model(model_name):
    """指定されたLLMモデルをロード"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.session_state.setdefault("load_messages", []).append(
            {"type": "info", "message": f"モデル '{model_name}' を {device} にロード中..."}
        )
        pipe = pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device
        )
        # トークナイザーのchat_templateを確認
        has_chat_template = hasattr(pipe.tokenizer, 'chat_template') and pipe.tokenizer.chat_template is not None
        st.session_state["load_messages"].append(
            {"type": "success", "message": f"モデル '{model_name}' のロードに成功しました。Chatテンプレート: {has_chat_template}"}
        )
        logging.info(f"モデル '{model_name}' のロードに成功しました。Chatテンプレート: {has_chat_template}")
        return pipe
    except Exception as e:
        st.session_state["load_messages"].append(
            {"type": "error", "message": f"モデル '{model_name}' のロードに失敗しました: {e}"}
        )
        logging.error(f"モデル '{model_name}' のロードに失敗しました: {e}")
        return None

def display_load_messages():
    """モデルロード時のメッセージを表示"""
    if "load_messages" in st.session_state:
        for msg in st.session_state["load_messages"]:
            if msg["type"] == "info":
                st.info(msg["message"])
            elif msg["type"] == "success":
                st.success(msg["message"])
            elif msg["type"] == "error":
                st.error(msg["message"])
        st.session_state["load_messages"] = []

def generate_response(pipe, user_question):
    """LLMを使用して質問に対する回答を生成"""
    if pipe is None:
        logging.error("モデルがロードされていません。")
        return "モデルがロードされていないため、回答を生成できません。", 0
    try:
        logging.debug(f"質問: {user_question}")
        start_time = time.time()
        
        # トークナイザーにchat_templateがあるか確認
        has_chat_template = hasattr(pipe.tokenizer, 'chat_template') and pipe.tokenizer.chat_template is not None
        logging.debug(f"Chatテンプレート使用: {has_chat_template}")

        if has_chat_template:
            # Gemma-2-2Bなど、チャットテンプレート対応モデルの場合
            messages = [{"role": "user", "content": user_question}]
            outputs = pipe(
                messages,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            logging.debug(f"モデル出力（チャット形式）: {outputs}")
            assistant_response = ""
            if outputs and isinstance(outputs, list) and outputs[0].get("generated_text"):
                generated_text = outputs[0]["generated_text"]
                if isinstance(generated_text, list) and generated_text:
                    last_message = generated_text[-1]
                    if last_message.get("role") == "assistant":
                        assistant_response = last_message.get("content", "").strip()
                elif isinstance(generated_text, str):
                    prompt_end = user_question
                    response_start_index = generated_text.find(prompt_end) + len(prompt_end)
                    assistant_response = generated_text[response_start_index:].strip()
                    if "<start_of_turn>model" in assistant_response:
                        assistant_response = assistant_response.split("<start_of_turn>model\n")[-1].strip()
        else:
            # XGLM-564Mなど、チャットテンプレート非対応モデルの場合
            outputs = pipe(
                user_question,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            logging.debug(f"モデル出力（プレーンテキスト）: {outputs}")
            assistant_response = ""
            if outputs and isinstance(outputs, list) and outputs[0].get("generated_text"):
                assistant_response = outputs[0]["generated_text"].strip()
                # プロンプト部分を除去（必要に応じて）
                if user_question in assistant_response:
                    assistant_response = assistant_response.replace(user_question, "").strip()

        if not assistant_response:
            st.warning("回答の抽出に失敗しました。")
            assistant_response = "回答を生成できませんでした。"
            logging.warning("回答の抽出に失敗しました。")
        response_time = time.time() - start_time
        logging.info(f"回答生成完了: 時間={response_time:.2f}s, 回答={assistant_response}")
        return assistant_response, response_time
    except Exception as e:
        st.error(f"回答生成中にエラーが発生しました: {e}")
        logging.error(f"回答生成中にエラーが発生しました: {e}")
        return f"エラー: {str(e)}", 0