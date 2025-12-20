#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import requests
from dotenv import load_dotenv

# 環境変数読み込み
load_dotenv()

class OllamaEmbedding:
    """Ollama Embedding APIクラス"""
    
    def __init__(self):
        base_url = os.getenv("OLLAMA_BASE_URL")
        if not base_url:
            raise ValueError("OLLAMA_BASE_URL が設定されていません")
        
        self.base_url = base_url
        self.model = "mxbai-embed-large"
    
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
            url = f"{self.base_url}/api/embeddings"
            
            payload = {
                "model": self.model,
                "prompt": text
            }
            
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            embedding = result.get('embedding')
            
            if not embedding:
                print("❌ Embeddingが取得できませんでした")
                return None
            
            # 次元数確認（1024次元）
            if len(embedding) != 1024:
                print(f"❌ 次元数エラー: {len(embedding)}次元（期待値: 1024次元）")
                return None
            
            return embedding
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Ollama API接続エラー: {e}")
            return None
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
        # Ollamaでは文書とクエリの区別がないため、get_embeddingと同じ
        return self.get_embedding(text)
    
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
    
    # ヘルスチェック接続
    def test_connection(self):
        """
        Ollama APIへの接続テスト
        Returns:
            bool: 接続成功時True、失敗時False
        """
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return True
        except:
            return False

# テスト実行
if __name__ == "__main__":
    print("=" * 50)
    print("Ollama Embedding APIテスト")
    print("=" * 50)
    
    embedder = OllamaEmbedding()
    
    # 接続テスト
    print(f"\nBase URL: {embedder.base_url}")
    if embedder.test_connection():
        print("✅ Ollama API接続成功")
    else:
        print("❌ Ollama API接続失敗")
        exit(1)
    
    # Embedding生成テスト
    test_text = "これはテスト用のテキストです"
    print(f"\nテキスト: {test_text}")
    print("Embedding生成中...")
    
    embedding = embedder.get_embedding(test_text)
    
    if embedding:
        print(f"✅ Embedding生成成功")
        print(f"  次元数: {len(embedding)}")
        print(f"  最初の5要素: {embedding[:5]}")
    else:
        print(f"❌ Embedding生成失敗")
        exit(1)
    
    print("\n全テスト成功！")