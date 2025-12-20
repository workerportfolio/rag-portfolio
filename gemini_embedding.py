#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import google.generativeai as genai
from dotenv import load_dotenv

# 環境変数読み込み
load_dotenv()

class GeminiEmbedding:
    """Google Gemini Embedding APIクラス"""
    
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY が設定されていません")
        
        genai.configure(api_key=api_key)
        self.model = "models/text-embedding-004"
    
    # ドキュメント(検索元データ)ベクトル化
    def get_embedding(self, text):
        """
        テキストをベクトル化
        Args:
            text: ベクトル化するテキスト
        Returns:
            ベクトル（リスト）、エラー時はNone
        """
        try:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"  # ドキュメント登録用 包括的・文書全体表現
            )
            
            # ベクトル取得
            embedding = result['embedding']

            # 次元数確認（768次元）
            if len(embedding) != 768:
                print(f"❌ 次元数エラー: {len(embedding)}次元（期待値: 768次元）")
            return None

            return embedding
            
        except Exception as e:
            print(f"❌ Embedding生成エラー: {e}")
            return None

    # 質問テキストベクトル化    
    def get_query_embedding(self, text):
        """
        検索クエリをベクトル化
        Args:
            text: ベクトル化する検索クエリ
        Returns:
            ベクトル（リスト）、エラー時はNone
        """
        try:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_query"  # クエリ用 キーワード・意味強調
            )
            
            # ベクトル取得
            embedding = result['embedding']
            
            return embedding
            
        except Exception as e:
            print(f"❌ Embedding生成エラー: {e}")
            return None
    
    #将来拡張検討用：現在未使用（元データ一括登録）
    def get_embeddings_batch(self, texts):
        """
        複数テキストを一括ベクトル化
        Args:
            texts: テキストのリスト
        Returns:
            ベクトルのリスト
        """
        try:
            embeddings = []
            for text in texts:
                embedding = self.get_embedding(text)
                if embedding:
                    embeddings.append(embedding)
            
            return embeddings if embeddings else None
            
        except Exception as e:
            print(f"❌ Embedding生成エラー: {e}")
            return None

# テスト実行
if __name__ == "__main__":
    print("=" * 50)
    print("Google Gemini Embedding APIテスト")
    print("=" * 50)
    
    embedder = GeminiEmbedding()
    
    # テストテキスト
    test_text = "Pythonは人気のプログラミング言語です。"
    
    print(f"\nテキスト: {test_text}")
    print("\nEmbedding生成中...")
    
    embedding = embedder.get_embedding(test_text)
    
    if embedding:
        print(f"✅ Embedding生成成功")
        print(f"ベクトル次元数: {len(embedding)}")
        print(f"最初の5要素: {embedding[:5]}")
    else:
        print("❌ Embedding生成失敗")