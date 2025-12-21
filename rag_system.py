#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from vector_store import VectorStore
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
        
        # Embeddingモデル設定
        self.embedding_model = embedding_model
        
        # VectorStoreとEmbedder初期化
        if embedding_model == 'google':
            print("\n Google Embedding（768次元）を使用")
            self.vector_store = VectorStore(model_type='google-768')
            self.embedder = GeminiEmbedding()
        elif embedding_model == 'ollama':
            print("\n Ollama Embedding（1024次元）を使用")
            self.vector_store = VectorStore(model_type='ollama-1024')
            self.embedder = OllamaEmbedding()
        else:
            raise ValueError(f"Invalid embedding_model: {embedding_model}")
        
        # LLM初期化
        self.use_local_llm = use_local_llm
        
        if use_local_llm:
            print("\n ローカルLLM（Ollama）を使用")
            self.llm = LocalLLM()
            if not self.llm.test_connection():
                raise Exception("ローカルLLMへの接続に失敗しました")
        else:
            print("\n クラウドLLM（Gemini）を使用")
            self.llm = CloudLLM()
            if not self.llm.test_connection():
                raise Exception("クラウドLLMへの接続に失敗しました")
        
        print("\n✅ RAGシステム初期化完了")
    
    def add_document(self, text: str, metadata: dict = None) -> int:
        """
        ドキュメント追加
        
        Args:
            text: ドキュメントのテキスト
            metadata: メタデータ（オプション）
        
        Returns:
            ドキュメントID
        """
        embedding = self.embedder.get_embedding(text)
        if not embedding:
            return None
        
        return self.vector_store.insert_document(text, embedding, metadata)
    
    def search(self, query_text: str, top_k: int = 3) -> dict:
        """
        類似ドキュメントを検索（デバッグ情報付き）
        
        Args:
            query_text: 検索クエリ
            top_k: 取得する上位N件
        
        Returns:
            検索結果の辞書（results, debug_info）
        """
        query_embedding = self.embedder.get_query_embedding(query_text)
        if not query_embedding:
            return {'results': [], 'debug_info': None}
        
        return self.vector_store.search_similar(query_embedding, top_k, self.embedding_model)
    
    def answer_question(self, question: str) -> dict:
        """
        質問に回答（デバッグ情報付き）
        
        Args:
            question: 質問文
        
        Returns:
            回答とデバッグ情報の辞書
        """
        # 関連ドキュメントを検索
        search_result = self.search(question, top_k=3)
        search_results = search_result['results']
        debug_info = search_result['debug_info']
        
        if not search_results:
            return {
                'answer': "関連するドキュメントが見つかりませんでした。",
                'debug_info': debug_info
            }
        
        # コンテキスト作成
        context = "\n\n".join([
            f"[ドキュメント{i+1}]\n{doc['text']}"
            for i, doc in enumerate(search_results)
        ])
        
        # プロンプト作成
        prompt = f"""以下のコンテキストに基づいて質問に回答してください。

コンテキスト:
{context}

質問: {question}

回答:"""
        
        # LLMで回答生成
        answer = self.llm.generate(prompt)
        
        if answer:
            return {
                'answer': answer,
                'debug_info': debug_info
            }
        else:
            return {
                'answer': "回答の生成に失敗しました。",
                'debug_info': debug_info
            }

# テスト実行
if __name__ == "__main__":
    import sys
    
    print("=" * 50)
    print("RAGシステム テストスクリプト")
    print("=" * 50)
    
    if len(sys.argv) < 2:
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
    try:
        rag = RAGSystem(use_local_llm=True, embedding_model='google')
    except Exception as e:
        print(f"❌ RAGシステム初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    if command == "add":
        text = " ".join(sys.argv[2:])
        print(f"\n追加するテキスト: {text}")
        doc_id = rag.add_document(text, {"source": "cli", "lang": "ja"})
        if doc_id:
            print(f"✅ ドキュメント追加完了 (ID: {doc_id})")
        else:
            print("❌ ドキュメント追加失敗")
        
    elif command == "search":
        query = " ".join(sys.argv[2:])
        print(f"\n検索クエリ: {query}")
        result = rag.search(query, top_k=3)
        
        print("\n検索結果:")
        for i, doc in enumerate(result['results'], 1):
            print(f"\n[{i}] ID: {doc['id']} (距離: {doc['distance']:.4f})")
            print(f"テキスト: {doc['text'][:100]}...")
        
    elif command == "ask":
        question = " ".join(sys.argv[2:])
        print(f"\n質問: {question}")
        result = rag.answer_question(question)
        
        print("\n" + "=" * 50)
        print("最終回答")
        print("=" * 50)
        print(result['answer'])
        
    else:
        print(f"❌ 不明なコマンド: {command}")
        sys.exit(1)