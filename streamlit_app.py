#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG検証・評価システム Streamlit WebUI
"""

import streamlit as st
from rag_system import RAGSystem
from db_connection import DatabaseConnection
import time

# ページ設定
st.set_page_config(
    page_title="RAG検証・評価システム",
    layout="wide"
)

# タイトル
st.title("RAG検証・評価システム - 質問応答UI")
st.markdown("---")

# サイドバー
# #############################################
st.sidebar.header("設定")

st.sidebar.markdown("### Embeddingモデル選択")

embedding_model = st.sidebar.radio(
    "使用Embeddingモデル",
    options=['google', 'ollama'],
    format_func=lambda x: "Google Embedding (768次元)" if x == 'google' else "Ollama Embedding (1024次元)",
    index=0,
    help="文書ベクトル化に使用するEmbeddingモデルを選択します"
)

# Embeddingモデル説明
with st.sidebar.expander("Embeddingモデルの違い"):
    st.markdown("""
    **Google Embedding (768次元)**
    - モデル: text-embedding-004
    - コスト: API料金（月100万トークン無料）
    - 精度: 高い
    - 速度: 速い
    ---------------------------
    **Ollama Embedding (1024次元)**
    - モデル: 
    - コスト: 無料（ローカル実行）
    - 精度: 中〜高
    - 速度: 中程度
    """)


st.sidebar.markdown("---")
st.sidebar.markdown("### LLMモデル選択")

use_local_llm = st.sidebar.radio(
    "使用LLM",
    options=[True, False],
    format_func=lambda x: "ローカルLLM (Ollama)" if x else "クラウドLLM (Gemini)",
    index=0,
    help="質問に応答するLLMを選択します"
)

# LLM説明
with st.sidebar.expander("LLMの違い"):
    st.markdown("""
    **ローカルLLM (Ollama)**
    - モデル: llama3.1:8b
    - コスト: 無料
    - 速度: やや遅い（5-10秒）
    - プライバシー: 高い
    - 精度: 中程度
    ---------------------------   
    **クラウドLLM (Gemini)**
    - モデル: gemini-2.0-flash-exp
    - コスト: API料金
    - 速度: 速い（2-5秒）
    - プライバシー: 外部API利用
    - 精度: 高い
    """)

st.sidebar.markdown("---")

# 現在の設定表示
st.sidebar.markdown("### 現在の設定")

# パターン判定
if embedding_model == 'google' and use_local_llm:
    pattern = "パターン1"
    pattern_desc = "Google Embedding + llama3.1"
elif embedding_model == 'google' and not use_local_llm:
    pattern = "パターン2"
    pattern_desc = "Google Embedding + Gemini"
elif embedding_model == 'ollama' and use_local_llm:
    pattern = "パターン3"
    pattern_desc = "Ollama Embedding + llama3.1"
else: #(embedding_model == 'ollama' and not use_local_llm:)
    pattern = "パターン4"
    pattern_desc = "Ollama Embedding + Gemini"

# 組み合わせパターン
st.sidebar.text(f"{pattern}")

# Embeddingモデル表示
embedding_info = "Google (768次元)" if embedding_model == 'google' else "Ollama (1024次元)"
st.sidebar.text(f"Embedding: {embedding_info}")

# LLMモデル表示
llm_info = "ローカルLLM" if use_local_llm else "クラウドLLM"
st.sidebar.text(f"LLM: {llm_info}")

# テーブル名表示
table_name = "documents_google_768" if embedding_model == 'google' else "documents_ollama_1024"
st.sidebar.text(f"テーブル: {table_name}")

st.sidebar.markdown("### システム情報")

# RAG検証・評価システム初期化（セッション状態で管理）
current_config = (use_local_llm, embedding_model)

if 'rag' not in st.session_state or st.session_state.get('config') != current_config:
    with st.spinner('RAG検証・評価システムを初期化中...'):
        try:
            st.session_state.rag = RAGSystem(
                use_local_llm=use_local_llm,
                embedding_model=embedding_model
            )
            st.session_state.config = current_config
            st.sidebar.success("✅ システム初期化完了")
        except Exception as e:
            st.sidebar.error(f"❌ 初期化エラー: {e}")
            import traceback
            st.sidebar.text(traceback.format_exc())
            st.stop()

# #############################################

# メイン画面
# #############################################
# セッション状態で会話履歴を管理
if 'history' not in st.session_state:
    st.session_state.history = []

# メインエリア: タブ構成
tab1, tab2, tab3 = st.tabs(["◆ 質問", "◆ 検索元データ追加", "◆ 統計情報"])

# タブ1: 質問
with tab1:
    st.header("◆ 質問してください")
    
    question = st.text_area(
        "質問入力:",
        placeholder="例: 登録データに基づいて◯◯を教えてください",
        height=200,
        key="question_input"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        submit_button = st.button("🔍 質問する", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🗑️ 履歴クリア", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    
    if submit_button:
        if question:
            with st.spinner('回答を生成中...'):
                start_time = time.time()
                
                try:
                    # 回答生成（デバッグ情報付き）
                    result = st.session_state.rag.answer_question(question)
                    answer = result['answer']
                    debug_info = result['debug_info']
                    elapsed_time = time.time() - start_time
                    
                    # 履歴に追加
                    st.session_state.history.append({
                        'question': question,
                        'answer': answer,
                        'time': elapsed_time,
                        'pattern': pattern,
                        'embedding': embedding_info,
                        'llm': llm_info,
                        'debug_info': debug_info
                    })
                    
                    # 回答表示
                    st.markdown("---")
                    st.subheader("回答")
                    st.write(answer)
                    st.caption(f"処理時間: {elapsed_time:.2f}秒 | {pattern}")
                    
                    # デバッグ情報表示
                    if debug_info:
                        with st.expander("デバッグ情報（検索詳細）", expanded=True):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("使用テーブル", debug_info['table_name'])
                                st.metric("Embeddingモデル", debug_info['embedding_model'])
                                st.metric("次元数", debug_info['embedding_dim'])
                            
                            with col2:
                                st.metric("top_k（raw）", debug_info['top_k_raw'])
                                st.metric("フィルタ後", debug_info['filtered_count'])
                                threshold_text = debug_info['threshold'] if debug_info['threshold'] else "なし"
                                st.metric("閾値", threshold_text)
                            
                            with col3:
                                discarded = len(debug_info['discarded_reasons'])
                                st.metric("切り捨て件数", discarded)
                            
                            # 検索結果一覧（rawベース）
                            st.markdown("### 検索結果（距離）")
                            st.caption("※ 距離が小さいほど類似度が高い（cosine distance）")
                            
                            for item in debug_info['results_raw']:
                                # rawは常に✅、切り捨てられたら⚠️
                                is_filtered = any(f['id'] == item['id'] for f in debug_info['results_filtered'])
                                status = "✅" if is_filtered else "⚠️"
                                
                                st.markdown(f"""
                                **{status} Rank {item['rank']} - ID: {item['id']}**
                                - 距離（distance）: `{item['distance']:.4f}` 
                                - テキスト: {item['text_preview']}...
                                """)
                            
                            # 切り捨て理由
                            if debug_info['discarded_reasons']:
                                st.markdown("### ⚠️ 閾値で切り捨てられた結果")
                                for reason in debug_info['discarded_reasons']:
                                    st.warning(f"ID {reason['id']}: {reason['reason']}")
                            
                            # 説明
                            st.info("""
                            **距離（distance）について:**
                            - pgvectorの `<=>` 演算子はcosine distanceを計算
                            - 値が小さいほど類似度が高い（0に近いほど似ている）
                            - 範囲: 0（完全一致）〜 2（正反対）
                            """)
                    
                except Exception as e:
                    st.error(f"❌ エラーが発生しました: {e}")
                    import traceback
                    st.text(traceback.format_exc())
        else:
            st.warning("⚠️ 質問を入力してください")
    
    # 会話履歴表示
    if st.session_state.history:
        st.markdown("---")
        st.subheader("会話履歴")
        
        for i, item in enumerate(reversed(st.session_state.history), 1):
            with st.expander(f"Q{len(st.session_state.history) - i + 1}: {item['question'][:50]}...", expanded=(i == 1)):
                st.markdown(f"**質問:** {item['question']}")
                st.markdown(f"**回答:** {item['answer']}")
                st.caption(f"{item['time']:.2f}秒 | {item.get('pattern', 'N/A')} | {item.get('embedding', 'N/A')} | {item.get('llm', 'N/A')}")

# タブ2: 検索元データ追加
with tab2:
    st.header("◆ データテキスト追加")
    
    st.markdown("RAG検証・評価システムに新規検索元データを追加します。")
    
    doc_text = st.text_area(
        "データテキスト:",
        placeholder="例: Pythonは、汎用プログラミング言語の一つです...",
        height=150,
        key="doc_text"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        category = st.text_input(
            "カテゴリ:",
            placeholder="例: Programming",
            key="category"
        )
    
    with col2:
        language = st.selectbox(
            "言語:",
            options=["ja", "en"],
            index=0,
            key="language"
        )
    
    if st.button("＋ ドキュメント追加", type="primary"):
        if doc_text:
            with st.spinner('ドキュメントを追加中...'):
                try:
                    metadata = {
                        "category": category if category else "未分類",
                        "lang": language
                    }
                    
                    doc_id = st.session_state.rag.add_document(doc_text, metadata)
                    
                    if doc_id:
                        st.success(f"✅ ドキュメントを追加しました（ID: {doc_id}）")
                    else:
                        st.error("❌ ドキュメントの追加に失敗しました")
                        
                except Exception as e:
                    st.error(f"❌ エラーが発生しました: {e}")
        else:
            st.warning("⚠️ ドキュメントのテキストを入力してください")

# タブ3: 統計情報
with tab3:
    st.header("◆ 統計情報")
    
    try:
        db = DatabaseConnection()
        if db.connect():
            # 現在のEmbeddingモデルに対応するテーブルからカウント
            current_table = "documents_google_768" if embedding_model == 'google' else "documents_ollama_1024"
            db.cursor.execute(f"SELECT COUNT(*) FROM {current_table};")
            doc_count = db.cursor.fetchone()[0]
            db.close()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("登録ドキュメント数", f"{doc_count}", help=f"テーブル: {current_table}")
            
            with col2:
                st.metric("会話履歴数", f"{len(st.session_state.history)}")
            
            with col3:
                llm_name = "ローカルLLM" if use_local_llm else "クラウドLLM"
                st.metric("現在のLLM", llm_name)
            
            if st.session_state.history:
                st.markdown("---")
                st.subheader("処理時間統計")
                
                times = [item['time'] for item in st.session_state.history]
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("平均処理時間", f"{avg_time:.2f}秒")
                
                with col2:
                    st.metric("最速処理時間", f"{min_time:.2f}秒")
                
                with col3:
                    st.metric("最遅処理時間", f"{max_time:.2f}秒")
        else:
            st.error("❌ データベースに接続できません")
            
    except Exception as e:
        st.error(f"❌ 統計情報の取得に失敗しました: {e}")
# #############################################

# フッター
st.markdown("---")
st.caption("RAG検証・評価システム v1.0 - ローカルLLM & クラウドLLM対応")