# RAGアプリケーション

## 概要
Retrieval-Augmented Generation (RAG) システムの基盤実装

## 構成
- VM1: PostgreSQL + Pgvector（ベクトルストア）
- VM2: RAGアプリ実行環境（Python）
- Windows: 開発環境（VSCode + Python）

## ファイル構成
```
rag-app/
├── .env                      # 環境変数（Git管理外）
├── .gitignore               # Git除外設定
├── db_connection.py         # DB接続モジュール
├── vector_store.py          # ベクトルストア管理
├── insert_test_data.py      # テストデータ投入
└── README.md                # このファイル
```

## セットアップ

### 1. 環境変数設定
`.env` ファイルを作成し、以下を設定:
```env
DB_HOST=192.168.100.10
DB_PORT=5432
DB_NAME=rag_db
DB_USER=rag_user
DB_PASSWORD=your_password
```

### 2. 必要ライブラリインストール
```bash
pip install -r requirements.txt
```

### 3. ベクトルストア初期化
```bash
python vector_store.py
```

### 4. テストデータ投入
```bash
python insert_test_data.py
```

## 次のステップ
- OpenAI API連携
- RAG検索機能実装
- ローカルLLM連携
