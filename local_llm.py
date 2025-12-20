#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ローカルLLM（Ollama）連携モジュール
"""

import ollama
import time
import os
import sys
from dotenv import load_dotenv

class LocalLLM:
    """ローカルLLM（Ollama）クライアント"""
    
    def __init__(self):
        """
        初期化
        Args:
            host: OllamaサーバーのURL
            model: 使用するモデル名
        """
        load_dotenv()
        
        host = os.getenv("OLLAMA_BASE_URL")
        if not host:
            raise ValueError("OLLAMA_BASE_URL が.envファイルに設定されていません")
        
        model = os.getenv("OLLAMA_MODEL")
        if not model:
            raise ValueError("OLLAMA_MODEL が.envファイルに設定されていません")

        self.host = host
        self.model = model
        
        # Ollamaクライアントの初期化
        self.client = ollama.Client(host=self.host)
        
        print(f"✅ LocalLLM初期化完了")
        print(f"   ホスト: {self.host}")
        print(f"   モデル: {self.model}")

    # LLM回答取得メソッド   
    def generate(self, prompt: str) -> str:
        """
        テキスト生成
        Args:
            prompt: プロンプトテキスト
        Returns:
            生成されたテキスト
        """
        try:
            start_time = time.time()
            
            # ローカルLLM APIを呼び出し プロンプト渡し
            response = self.client.generate(
                model=self.model,
                prompt=prompt
            )
            
            elapsed_time = time.time() - start_time
            
            print(f"✅ ローカルLLM応答完了")
            print(f"   処理時間: {elapsed_time:.2f}秒")
            return response['response'] #ローカルLLM回答取得 
        except Exception as e:
            print(f"❌ ローカルLLM応答エラー: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # ヘルスチェック接続
    def test_connection(self) -> bool:
        """
        Ollamaサーバーへの接続テスト
        Returns:
            接続成功ならTrue
        """
        try:
            models = self.client.list()
            
            if hasattr(models, 'models'):
                model_list = [m.model for m in models.models]
            else:
                model_list = []
            
            print(f"✅ Ollamaサーバー接続成功")
            
            # 使用予定モデルが存在するか確認
            if self.model in model_list:
                print(f"   ✅ モデル '{self.model}' が利用可能")
                return True
            else:
                print(f"   ⚠️ モデル '{self.model}' が見つかりません")
                print(f"   利用可能: {model_list}")
                return False
                
        except Exception as e:
            print(f"❌ Ollamaサーバー接続エラー: {e}")
            import traceback
            traceback.print_exc()
            return False


# テスト用
if __name__ == "__main__":
    print("=" * 50)
    print("LocalLLM（Ollama）テスト")
    print("=" * 50)
    
    # 初期化
    llm = LocalLLM()
    
    # 接続テスト
    print("\n--- 接続テスト ---")
    if not llm.test_connection():
        print("❌ 接続テスト失敗")
        exit(1)
    
    # テキスト生成テスト
    print("\n--- テキスト生成テスト ---")
    
    prompt = "機械学習とは何ですか？簡潔に説明してください。"
    
    answer = llm.generate(prompt)
    
    if answer:
        print(f"✅ 生成成功")
        print(f"\n【生成された回答】")
        print(answer)
    else:
        print(f"❌ 生成エラー")
    
    print("\n" + "=" * 50)