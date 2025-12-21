# RAG検証・評価システム

## 概要
複数のEmbeddingモデルとLLMの組み合わせを比較評価できるRAGシステム。
PostgreSQL + pgvector を使用し、4パターンの構成で検索挙動および回答品質の違いを検証。

## プロジェクトの目的
- 異なるEmbeddingモデル（Google / Ollama）の性能比較
- 異なるLLM（Gemini 2.0 Flash / llama3.1）の回答品質比較
- RAGシステムにおける構成要素と実装パターンの理解

## 技術スタック
- **バックエンド**: Python 3.11
- **データベース**: PostgreSQL 17 + pgvector 0.8.1
- **フロントエンド**: Streamlit
- **Embedding**: Google Embedding API (768次元) / Ollama mxbai-embed-large (1024次元)
- **LLM**: Google Gemini 2.0 Flash / Ollama llama3.1:8b
- **インフラ**: Hyper-V (3VM構成)

## システム構成
```
[VM1: PostgreSQL + pgvector] ⇄ [VM2: Python App + Streamlit] ⇄ [VM3: Ollama LLM Server]
```

- **VM1 (DBServer)**: PostgreSQL + pgvector
  - documents_google_768 テーブル
  - documents_ollama_1024 テーブル
- **VM2 (APIServer)**: Python Application + Streamlit WebUI
- **VM3 (LLMServer)**: Ollama LLM Server
  - llama3.1:8b-instruct-q4_K_M
  - mxbai-embed-large

## 主要機能

### 1. 4パターンの性能比較機能
- **パターン1**: Google Embedding (768次元) + llama3.1
- **パターン2**: Google Embedding (768次元) + Gemini
- **パターン3**: Ollama Embedding (1024次元) + llama3.1
- **パターン4**: Ollama Embedding (1024次元) + Gemini

UIでパターンを簡単に切り替えて、同じ質問に対する検索精度と回答品質を比較可能。

### 2. 検索精度の可視化機能
- 検索結果の距離（cosine distance）表示
- 使用テーブル・Embeddingモデルの表示
- top_k件数、フィルタ後の件数表示
- 各ドキュメントのランク付け

### 3. ドキュメント管理機能
- メタデータ付きドキュメント登録
  - カテゴリ分類
  - 言語設定（ja/en）
- 統計情報表示

## 技術的な成果

### 1. pgvectorインデックス特性理解と選定判断

**背景:**
IVFFLATインデックスの特性理解が不十分な状態で
`ORDER BY ... LIMIT` を用いたベクトル検索を実装した結果、
意図しない検索結果0件が発生。

**対応・検証プロセス:**
1. PostgreSQL上でクエリを直接実行し、挙動を詳細に検証
2. インデックス有無・検索条件・LIMIT指定の差分を比較
3. IVFFLATがクラスタリングを前提とした近似検索インデックスであり、
   データ量が少ない状態やパラメータ設定・クエリ条件によっては
   期待した結果が得られないケースがあることを理解
4. 本プロジェクトの要件（安定した検索結果取得）に適した
   HNSWインデックスへの移行を判断・実施

**成果:**
- 検索が安定して動作するRAG構成を実現
- pgvectorにおけるインデックス特性を理解した上での
  技術選定・設計判断の重要性を実践的に習得

### 2. 異なる次元数のEmbeddingへの対応
- Google Embedding: 768次元
- Ollama Embedding: 1024次元
- PostgreSQLのVECTOR型は次元数固定のため、別テーブルで管理
- アプリケーションレイヤーで動的に切り替える設計

**背景:**
異なる次元数のEmbeddingによる性能差を評価対象とした。

**設計判断:**
- PostgreSQLのVECTOR型は次元数固定の制約を確認
- 別テーブル管理によるアーキテクチャを選択
- アプリケーションレイヤーで4パターンを動的に切り替え可能な設計

**成果:**
単一UIから4パターン（Embedding/LLM）を切り替え可能な比較基盤を実現

### 3. Embeddingモデルの検索傾向の検証と設計上の示唆

**背景:**
抽象度の高い質問文（例：「仮想化技術にはどのようなものがある？」）に対し、
期待した具体的技術（例：Hyper-V）が検索結果の最上位に返らないケースを確認。

**検証・理解:**
- Embedding検索では、質問文とドキュメントの
  「意味的抽象度の近さ」が類似度計算に影響することを確認
- 抽象的な質問に対しては、概念説明系のドキュメントが上位に来やすい傾向があることを理解
- Embeddingモデルによって順位付け傾向の差が顕著であることを確認
- コサイン距離の分布から、検索結果自体は破綻しておらず、
  順位付けの特性によるものであると判断

**得られた知見:**
- 質問文の具体性
- ドキュメント粒度の揃え方
- top_k の設定が検索品質に大きく影響することを確認

**成果:**
- Embeddingモデルの特性を前提としたRAG設計の重要性を理解
- 検索精度はモデル性能だけでなく、
  入力設計・データ設計に依存することを実データで検証

## 開発プロセスと学習

### AI活用について
本プロジェクトは、ChatGPT, Claude を積極的に活用して開発。

**AIが支援した部分:**
- Pythonコードの実装・デバッグ
- 複数の実装パターン・ベストプラクティス案の提示
- エラー解決の支援

**自分で実施・判断した部分:**
- システム全体のアーキテクチャ設計（3VM構成）
- 提示されたベストプラクティスを比較検討し、
  本プロジェクトに適合する設計・実装方針を選定
- 要件定義から完成までのプロジェクト管理
- AIが支援したコードに対し、
  処理の重複・責務分離・モジュール境界の観点で検証を行い、
  保守性・拡張性を考慮した構成へ設計判断を行った
- pgvector IVFFLATインデックスの特性理解不足に起因する検索0件事象について、
  データベースレベルでの検証・切り分けを行い、
  プロジェクト要件に適したHNSWインデックスへの移行を判断
- Embeddingモデルの性能比較設計
- PostgreSQL + pgvectorのセットアップ
- Hyper-V仮想環境の構築・ネットワーク設定

### なぜAIを活用したか
1. **AIの利用価値の確認**: AIの業務利用への有用性確認
2. **開発速度の向上**: 3ヶ月の予定を1ヶ月で完成
3. **ベストプラクティスの学習**: AIとの対話を通じて最新の実装パターンを習得
4. **本質的な問題解決に集中**: コーディング作業を効率化し、システム設計や問題分析に時間を投資

### 学んだこと
AIとの協働開発を通じて、以下を深く理解しました:
- RAGアーキテクチャの構成要素と実装パターン
- ベクトルデータベースのインデックス戦略（IVFFLAT vs HNSW）
- 複数のEmbeddingモデルの特性とトレードオフ
- 効果的なプロンプトエンジニアリング手法
- 問題の切り分けとデバッグの体系的なアプローチ
- 各生成AIの得意・不得意分野の理解(使い分けの重要性)

## ファイル構成
```
rag-app/
├── rag_system.py           # RAGシステム本体
├── cloud_llm.py            # クラウドLLM (Gemini)
├── local_llm.py            # ローカルLLM (Ollama)
├── gemini_embedding.py     # Google Embedding
├── ollama_embedding.py     # Ollama Embedding
├── db_connection.py        # データベース接続
├── vector_store.py         # ベクトルストア管理
├── streamlit_app.py        # WebUI
├── requirements.txt        # 依存関係
├── .env                    # 環境変数（Git管理外）
└── README.md               # このファイル
```

## セットアップ

### 1. 環境変数設定
`.env` ファイルを作成:
```env
DB_HOST=192.168.xxx.xxx
DB_PORT=5432
DB_NAME=rag_db
DB_USER=rag_user
DB_PASSWORD=your_password
GEMINI_API_KEY=your_api_key
```

### 2. 依存関係インストール
```bash
pip install -r requirements.txt
```

### 3. Streamlit起動
```bash
streamlit run streamlit_app.py

※ VM上で起動し、別端末からアクセスする場合は
`--server.address 0.0.0.0` を指定。
```

## デモ動画
**4パターンの比較デモ（1分）:**
[デモ動画を見る](demo.mp4)  
※ 本デモは UI 操作の紹介ではなく、
Embedding / LLM 切替による RAG 挙動の違いを示すことを目的としています。

## 今後の拡張予定
- CSV一括登録機能（構造化データの一括投入）
- ハイブリッド検索（ベクトル + キーワード検索）
- メタデータフィルタリング機能（言語別、カテゴリ別検索）
- Embeddingモデルのファインチューニング
- 検索精度の定量評価
- 次元数の違いが検索精度・回答品質に与える影響検証
- ドキュメントへの添付ファイル対応（PDF / 画像）
