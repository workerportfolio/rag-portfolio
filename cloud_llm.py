#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
クラウドLLM（Gemini）連携モジュール
"""

import google.generativeai as genai
import os
from dotenv import load_dotenv


class CloudLLM:
    """クラウドLLM（Gemini）クライアント"""
    
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        初期化
        
        Args:
            model_name: 使用するGeminiモデル名
        """
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY が.envファイルに設定されていません")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        print(f"✅ CloudLLM初期化完了")
        print(f"   モデル: {model_name}")
    
    def generate(self, prompt: str) -> str:
        """
        テキスト生成
        
        Args:
            prompt: プロンプトテキスト
        
        Returns:
            生成されたテキスト
        """
        try:
            response = self.model.generate_content(prompt)
            print(f"✅ クラウドLLM応答完了")
            return response.text
        except Exception as e:
            print(f"❌ クラウドLLM応答エラー: {e}")
            import traceback
            traceback.print_exc()
            return None


# テスト用
if __name__ == "__main__":
    print("=" * 50)
    print("CloudLLM（Gemini）テスト")
    print("=" * 50)
    
    llm = CloudLLM()
    
    prompt = "機械学習とは何ですか？簡潔に説明してください。"
    
    answer = llm.generate(prompt)
    
    if answer:
        print(f"\n【生成された回答】")
        print(answer)
    
    print("\n" + "=" * 50)