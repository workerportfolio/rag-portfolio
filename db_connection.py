#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import psycopg2
from psycopg2.extensions import register_adapter, AsIs
from dotenv import load_dotenv

# 環境変数読み込み
load_dotenv()

# リストをPostgreSQL vector型に変換する関数(psycopg2 特性のため必要処理)
def adapt_list_to_vector(lst):
    """
    PythonのリストをPostgreSQLのvector型文字列に変換
    例: [1.0, 2.0, 3.0] -> "'[1.0,2.0,3.0]'"
    """
    vector_str = '[' + ','.join(map(str, lst)) + ']'
    return AsIs("'" + vector_str + "'")

# psycopg2にリスト→vector変換を登録
register_adapter(list, adapt_list_to_vector)

class DatabaseConnection:
    """データベース接続クラス"""
    
    def __init__(self):
        self.connection = None
        self.cursor = None
        
        # 環境変数から接続情報取得
        self.host = os.getenv("DB_HOST")
        self.port = os.getenv("DB_PORT")
        self.database = os.getenv("DB_NAME")
        self.user = os.getenv("DB_USER")
        self.password = os.getenv("DB_PASSWORD")
    
    # データベース接続処理
    def connect(self):
        """データベース接続"""
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            self.cursor = self.connection.cursor()
            print(f"✅ データベース接続成功: {self.database}")
            return True
        except Exception as e:
            print(f"❌ データベース接続エラー: {e}")
            return False
    
    # データベース切断処理
    def close(self):
        """データベース接続を閉じる"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        print("✅ データベース接続を閉じました")
    
    # クエリ実行処理
    def execute(self, query, params=None):
        """クエリ実行"""
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            
            # SELECT文 または RETURNING句がある場合はfetchall()
            query_upper = query.strip().upper()
            
            if query_upper.startswith('SELECT') or 'RETURNING' in query_upper:
                result = self.cursor.fetchall()
                return result
            else:
                return True
                
        except Exception as e:
            print(f"❌ クエリ実行エラー: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # コミット処理
    def commit(self):
        """コミット"""
        if self.connection:
            self.connection.commit()

# テスト実行
if __name__ == "__main__":
    db = DatabaseConnection()

    if db.connect():
        # PostgreSQLバージョン確認
        db.execute("SELECT version();")
        result = db.cursor.fetchone()
        print(f"\nPostgreSQL: {result[0]}")

        # Pgvectorテスト
        db.execute("SELECT '[1,2,3]'::vector;")
        result = db.cursor.fetchone()
        print(f"Pgvector: {result[0]}")
        
        db.close()

