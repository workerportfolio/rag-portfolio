#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ローカルLLM（Ollama）連携モジュール
"""

import ollama
import time


class LocalLLM:
    """ローカルLLM（Ollama）クライアント"""
    
    def __init__(
        self,
        host: str = "http://192.168.100.30:11434",
        model: str = "llama3.1:8b-instruct-q4_K_M"
    ):
        """
        初期化
        
        Args:
            host: OllamaサーバーのURL
            model: 使用するモデル名
        """
        self.host = host
        self.model = model
        
        # Ollamaクライアントの初期化
        self.client = ollama.Client(host=self.host)
        
        print(f"✅ LocalLLM初期化完了")
        print(f"   ホスト: {self.host}")
        print(f"   モデル: {self.model}")
    
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
            
            # Ollama APIを呼び出し
            response = self.client.generate(
                model=self.model,
                prompt=prompt
            )
            
            elapsed_time = time.time() - start_time
            
            print(f"✅ ローカルLLM応答完了")
            print(f"   処理時間: {elapsed_time:.2f}秒")
            
            return response['response']
            
        except Exception as e:
            print(f"❌ ローカルLLM応答エラー: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_connection(self) -> bool:
        """
        Ollamaサーバーへの接続テスト
        
        Returns:
            接続成功ならTrue
        """
        try:
            # モデル一覧を取得して接続確認
            models = self.client.list()

            print("=== Ollama list() 生データ ===")
            print(models)

            # パターン1: models.models という属性を持つオブジェクト
            if hasattr(models, "models"):
                raw_models = models.models
            # パターン2: そのままリスト
            elif isinstance(models, list):
                raw_models = models
            # パターン3: dict 形式（将来の仕様変更対策）
            elif isinstance(models, dict):
                raw_models = models.get("models", [])
            else:
                raw_models = []

            print(f"✅ Ollamaサーバー接続成功")
            print(f"   利用可能なモデル数: {len(raw_models)}")

            model_names = []
            for m in raw_models:
                # Modelオブジェクトの場合
                if hasattr(m, "name"):
                    model_names.append(m.name)
                elif hasattr(m, "model"):
                    model_names.append(m.model)
                # dictの場合
                elif isinstance(m, dict):
                    n = m.get("name") or m.get("model")
                    if n:
                        model_names.append(n)

            print(f"   利用可能なモデル: {model_names}")

            if self.model in model_names:
                print(f"   ✅ モデル '{self.model}' が利用可能")
                return True
            else:
                print(f"   ⚠️ モデル '{self.model}' が見つかりません")
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