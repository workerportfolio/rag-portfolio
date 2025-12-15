#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from db_connection import DatabaseConnection
from gemini_embedding import GeminiEmbedding
from ollama_embedding import OllamaEmbedding
from local_llm import LocalLLM
from cloud_llm import CloudLLM

class RAGSystem:
    """RAGシステム統合クラス"""
    
    def __init__(self, use_local_llm: bool = True, embedding_model: str = 'google'):
        """
        初期化
        
        Args:
            use_local_llm: Trueならローカル、FalseならGemini
            embedding_model: 'google' または 'ollama'
        """
        print("=" * 50)
        print("RAGシステム初期化")
        print("=" * 50)
        
        self.db = DatabaseConnection()
        
        # Embeddingモデル初期化
        self.embedding_model = embedding_model
        
        if embedding_model == 'google':
            print("\n📊 Google Embedding（768次元）を使用")
            self.embedder = GeminiEmbedding()
            self.table_name = 'documents_google_768'
        elif embedding_model == 'ollama':
            print("\n📊 Ollama Embedding（1024次元）を使用")
            self.embedder = OllamaEmbedding()
            self.table_name = 'documents_ollama_1024'
        else:
            raise ValueError(f"Invalid embedding_model: {embedding_model}")
        
        # LLM初期化
        self.use_local_llm = use_local_llm
        
        if use_local_llm:
            print("\n🤖 ローカルLLM（Ollama）を使用")
            self.llm = LocalLLM(
                host="http://192.168.100.30:11434",
                model="llama3.1:8b-instruct-q4_K_M"
            )
            # 接続テスト
            if not self.llm.test_connection():
                raise Exception("ローカルLLMへの接続に失敗しました")
        else:
            print("\n☁️ クラウドLLM（Gemini）を使用")
            self.llm = CloudLLM()
        
        print("\n✅ RAGシステム初期化完了")
    
    def add_document(self, text: str, metadata: dict = None) -> int:
        """
        ドキュメントを追加
        
        Args:
            text: ドキュメントのテキスト
            metadata: メタデータ（オプション）
        
        Returns:
            ドキュメントID
        """
        print("\n" + "=" * 50)
        print("ドキュメント追加")
        print("=" * 50)
        
        if not self.db.connect():
            return None
        
        try:
            # テキストをベクトル化
            print(f"\n[1] テキストをベクトル化中...")
            embedding = self.embedder.get_embedding(text)
            
            if not embedding:
                print("❌ ベクトル化に失敗しました")
                return None
            
            print(f"✅ ベクトル化完了（次元数: {len(embedding)}）")
            
            # データベースに保存
            print(f"\n[2] データベースに保存中...")
            
            insert_query = f"""
            INSERT INTO {self.table_name} (document_text, embedding, metadata)
            VALUES (%s, %s, %s)
            RETURNING id;
            """
            
            result = self.db.execute(
                insert_query,
                (text, embedding, json.dumps(metadata) if metadata else None)
            )
            
            self.db.commit()
            
            if result:
                doc_id = result[0][0]
                print(f"✅ ドキュメント追加完了（ID: {doc_id}）")
                print("=" * 50)
                return doc_id
            else:
                print("❌ ドキュメント追加に失敗しました")
                return None
                
        except Exception as e:
            print(f"❌ エラー: {e}")
            return None
        finally:
            self.db.close()
    
    def search(self, query_text: str, top_k: int = 3) -> dict:
        """
        類似ドキュメントを検索（デバッグ情報付き）
        
        Args:
            query_text: 検索クエリ
            top_k: 取得する上位N件
        
        Returns:
            検索結果の辞書（results, debug_info）
        """
        print("\n" + "=" * 50)
        print("類似ドキュメント検索")
        print("=" * 50)
        
        if not self.db.connect():
            return {'results': [], 'debug_info': None}
        
        try:
            # クエリをベクトル化
            print(f"\n[1] クエリをベクトル化中...")
            print(f"クエリ: {query_text}")
            
            query_embedding = self.embedder.get_query_embedding(query_text)
            
            if not query_embedding:
                print("❌ クエリのベクトル化に失敗しました")
                return {'results': [], 'debug_info': None}
            
            print(f"✅ ベクトル化完了（次元数: {len(query_embedding)}）")
            
            # ベクトル検索
            print(f"\n[2] ベクトル検索中（上位{top_k}件）...")
            
            # ベクトルをPostgreSQL形式に変換
            vector_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            search_query = f"""
            SELECT 
                id,
                document_text,
                metadata,
                embedding <=> '{vector_str}'::vector({len(query_embedding)}) as distance
            FROM {self.table_name}
            ORDER BY embedding <=> '{vector_str}'::vector({len(query_embedding)})
            LIMIT %s;
            """
            
            self.db.cursor.execute(search_query, (top_k,))
            results_raw = self.db.cursor.fetchall()
            
            print(f"✅ {len(results_raw)}件のドキュメントを取得")
            
            # デバッグ情報を構築
            debug_info = {
                'table_name': self.table_name,
                'embedding_model': self.embedding_model,
                'embedding_dim': len(query_embedding),
                'top_k_raw': len(results_raw),
                'threshold': None,
                'results_raw': [],
                'results_filtered': [],
                'filtered_count': 0,
                'discarded_reasons': []
            }
            
            # 生データを記録
            for i, (doc_id, text, metadata, distance) in enumerate(results_raw, 1):
                debug_info['results_raw'].append({
                    'rank': i,
                    'id': doc_id,
                    'distance': distance,
                    'text_preview': text[:100]
                })
                print(f"\n[{i}] ID: {doc_id} | 距離: {distance:.4f}")
                print(f"    テキスト: {text[:100]}...")
            
            # フィルタリング（閾値があれば適用、現在はなし）
            threshold = None
            
            results_filtered = []
            for doc_id, text, metadata, distance in results_raw:
                if threshold is None or distance <= threshold:
                    results_filtered.append({
                        'id': doc_id,
                        'text': text,
                        'metadata': metadata,
                        'distance': distance
                    })
                    debug_info['results_filtered'].append({
                        'id': doc_id,
                        'distance': distance
                    })
                else:
                    debug_info['discarded_reasons'].append({
                        'id': doc_id,
                        'reason': f'距離 {distance:.4f} > 閾値 {threshold}',
                        'distance': distance
                    })
            
            debug_info['filtered_count'] = len(results_filtered)
            
            print(f"\n✅ フィルタ後: {len(results_filtered)}件")
            print("=" * 50)
            
            return {
                'results': results_filtered,
                'debug_info': debug_info
            }
            
        except Exception as e:
            print(f"❌ エラー: {e}")
            import traceback
            traceback.print_exc()
            return {'results': [], 'debug_info': None}
        finally:
            self.db.close()

    
    def answer_question(self, question: str) -> dict:
        """
        質問に回答（デバッグ情報付き）
        
        Args:
            question: 質問文
        
        Returns:
            回答とデバッグ情報の辞書
        """
        print("\n" + "=" * 50)
        print("質問応答")
        print("=" * 50)
        print(f"質問: {question}")
        
        # 関連ドキュメントを検索
        print("\n" + "-" * 50)
        print("Step 1: 関連ドキュメント検索")
        print("-" * 50)
        
        search_result = self.search(question, top_k=3)
        search_results = search_result['results']
        debug_info = search_result['debug_info']
        
        if not search_results:
            return {
                'answer': "関連するドキュメントが見つかりませんでした。",
                'debug_info': debug_info
            }
        
        # コンテキスト作成
        print("\n" + "-" * 50)
        print("Step 2: コンテキスト作成")
        print("-" * 50)
        
        context = "\n\n".join([
            f"[ドキュメント{i+1}]\n{doc['text']}"
            for i, doc in enumerate(search_results)
        ])
        
        print(f"コンテキスト文字数: {len(context)}文字")
        
        # プロンプト作成
        prompt = f"""以下のコンテキストに基づいて質問に回答してください。

    コンテキスト:
    {context}

    質問: {question}

    回答:"""
        
        # LLMで回答生成
        print("\n" + "-" * 50)
        print("Step 3: LLMで回答生成")
        print("-" * 50)
        
        answer = self.llm.generate(prompt)
        
        if answer:
            print("\n✅ 回答生成完了")
            print("=" * 50)
            return {
                'answer': answer,
                'debug_info': debug_info
            }
        else:
            print("\n❌ 回答生成に失敗しました")
            print("=" * 50)
            return {
                'answer': "回答の生成に失敗しました。",
                'debug_info': debug_info
            }

# テスト実行
if __name__ == "__main__":
    import sys
    
    # 使用方法を表示
    if len(sys.argv) < 2:
        print("=" * 50)
        print("RAGシステム テストスクリプト")
        print("=" * 50)
        print("\n使用方法:")
        print("  python rag_system.py add <テキスト>")
        print("  python rag_system.py search <クエリ>")
        print("  python rag_system.py ask <質問>")
        print("\n例:")
        print('  python rag_system.py add "Pythonは汎用プログラミング言語です"')
        print('  python rag_system.py search "プログラミング"')
        print('  python rag_system.py ask "Pythonとは何ですか？"')
        sys.exit(1)
    
    command = sys.argv[1]
    
    # RAGシステム初期化
    rag = RAGSystem(use_local_llm=True)
    
    if command == "add":
        # ドキュメント追加
        text = " ".join(sys.argv[2:])
        doc_id = rag.add_document(text, {"source": "cli", "lang": "ja"})
        
    elif command == "search":
        # 検索
        query = " ".join(sys.argv[2:])
        results = rag.search(query, top_k=3)
        
    elif command == "ask":
        # 質問応答
        question = " ".join(sys.argv[2:])
        answer = rag.answer_question(question)
        print("\n" + "=" * 50)
        print("最終回答")
        print("=" * 50)
        print(answer)
        
    else:
        print(f"❌ 不明なコマンド: {command}")
        sys.exit(1)

