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

### 1. pgvectorインデックス問題の発見と解決
**問題:** IVFFLATインデックス使用時、`ORDER BY + LIMIT` を含むクエリで検索結果が0件になるバグを発見。

**解決プロセス:**
1. PostgreSQLで直接SQL実行して原因を特定
2. インデックスなしでは正常動作を確認
3. HNSWインデックスへの移行で解決

**成果:** システムが正常動作し、インデックス戦略の重要性を実証

### 2. 異なる次元数のEmbeddingへの対応
- Google Embedding: 768次元
- Ollama Embedding: 1024次元
- PostgreSQLのVECTOR型は次元数固定のため、別テーブルで管理
- アプリケーションレイヤーで動的に切り替える設計

### 3. ハイブリッド環境の構築
- Windows開発環境でコーディング・テスト
- SSH/SCP経由でLinux VMにデプロイ
- 3VM構成で役割を分離（DB / App / LLM）

## 開発プロセスと学習

### AI活用について
本プロジェクトは、ChatGPT, Claude を積極的に活用して開発。

**AIが支援した部分:**
- Pythonコードの実装・デバッグ
- ベストプラクティスの提案
- エラー解決の支援

**自分で実施・判断した部分:**
- システム全体のアーキテクチャ設計（3VM構成）
- PostgreSQL + pgvectorのセットアップ
- Hyper-V仮想環境の構築・ネットワーク設定
- **IVFFLATインデックスバグの発見と解決プロセス**
  - 検索結果が0件になる問題の特定
  - データベース直接操作での原因究明
  - HNSWインデックスへの移行判断
- Embeddingモデルの性能比較設計
- 要件定義から完成までのプロジェクト管理

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
- 検索精度の定量評価（Precision, Recall）
- pgvectorのインデックス違いによる戦略（IVFFLAT / HNSW）の検証
- 次元数の違いが検索精度・回答品質に与える影響検証
- ドキュメントへの添付ファイル対応（PDF / 画像）
