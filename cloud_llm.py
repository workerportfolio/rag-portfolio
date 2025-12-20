#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
クラウドLLM（Gemini）連携モジュール
"""

import google.generativeai as genai
import os
import time
import sys
from dotenv import load_dotenv


class CloudLLM:
    """クラウドLLM（Gemini）クライアント"""
    
    def __init__(self):
        """
        初期化
        Args:
            model: 使用するGeminiモデル名
        """
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY が.envファイルに設定されていません")
        
        model = os.getenv('GEMINI_MODEL')
        if not model:
            raise ValueError("GEMINI_MODEL が.envファイルに設定されていません")
        
        genai.configure(api_key=api_key)
        self.model = model
        self.client = genai.GenerativeModel(model)
        
        print(f"✅ CloudLLM初期化完了")
        print(f"   モデル: {self.model}")
    
    # LLM回答取得メソッド
    def generate(self, prompt: str) -> str:
        """
        Gemini APIに問い合わせて回答を取得
        Args:
            prompt: 送信するプロンプトテキスト
        Returns:
            Gemini APIから返された生成テキスト
            エラー時はNoneを返す
        """
        try:
            start_time = time.time()
            
            #クラウドLLMにプロンプト渡し
            response = self.client.generate_content(prompt)

            elapsed_time = time.time() - start_time

            print(f"✅ クラウドLLM応答完了")
            print(f"処理時間: {elapsed_time:.2f}秒")
            return response.text #クラウドLLM回答取得
        except Exception as e:
            print(f"❌ クラウドLLM応答エラー: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # ヘルスチェック接続
    def test_connection(self) -> bool:
        """
        Gemini APIへの接続テスト
        Returns:
            接続成功ならTrue
        """
        try:
            models = genai.list_models()
            model_list = [m.name for m in models]
            
            print(f"✅ Gemini API接続成功")
            
            # 部分一致で確認
            available = any(self.model in m for m in model_list)
            
            if available:
                print(f"   ✅ モデル '{self.model}' が利用可能")
                return True
            else:
                print(f"   ⚠️ モデル '{self.model}' が見つかりません")
                print(f"   利用可能: {model_list}")
                return False
                
        except Exception as e:
            print(f"❌ Gemini API接続エラー: {e}")
            import traceback
            traceback.print_exc()
            return False

# テスト用
if __name__ == "__main__":
    print("=" * 50)
    print("CloudLLM（Gemini）テスト")
    print("=" * 50)
    
    llm = CloudLLM()

    # 接続テスト追加
    print("\n--- 接続テスト ---")
    if not llm.test_connection():
        print("❌ 接続テスト失敗")
        sys.exit(1)

    print("\n--- テキスト生成テスト ---")  
    prompt = "機械学習とは何ですか？簡潔に説明してください。"
    
    answer = llm.generate(prompt)
    
    if answer:
        print(f"\n【生成された回答】")
        print(answer)
    
    print("\n" + "=" * 50)