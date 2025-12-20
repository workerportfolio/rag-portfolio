#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from db_connection import DatabaseConnection

class VectorStore:
    """ベクトルストア管理クラス（複数テーブル対応）"""
    
    # テーブル設定
    TABLE_CONFIG = {
        'google-768': {
            'table_name': 'documents_google_768',
            'embedding_dim': 768,
            'vector_type': 'vector(768)'
        },
        'ollama-1024': {
            'table_name': 'documents_ollama_1024',
            'embedding_dim': 1024,
            'vector_type': 'vector(1024)'
        }
    }
    
    def __init__(self, model_type='google-768'):
        """
        初期化
        
        Args:
            model_type: 'google-768' または 'ollama-1024'
        """
        if model_type not in self.TABLE_CONFIG:
            raise ValueError(f"Invalid model_type: {model_type}. Use 'google-768' or 'ollama-1024'")
        
        self.model_type = model_type
        config = self.TABLE_CONFIG[model_type]
        self.table_name = config['table_name']
        self.embedding_dim = config['embedding_dim']
        self.vector_type = config['vector_type']
        
        self.db = DatabaseConnection()

    # 将来的にはテーブルへの挿入、テーブル削除、テーブル更新処理も本pyに集約

    # テーブル作成処理
    def create_table(self):
        """ベクトル検索用テーブル作成"""
        print("\n" + "=" * 50)
        print(f"ベクトルストアテーブル作成: {self.table_name}")
        print("=" * 50)
        
        if not self.db.connect():
            return False
        
        try:
            # テーブル作成
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id BIGSERIAL PRIMARY KEY,
                document_text TEXT NOT NULL,
                embedding {self.vector_type},
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
            
            print("\n[1] テーブル作成中...")
            self.db.execute(create_table_query)
            self.db.commit()
            print(f"✅ {self.table_name}テーブル作成完了")
            
            # インデックス作成（ベクトル検索高速化）
            create_index_query = f"""
            CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx 
            ON {self.table_name} 
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
            """
            
            print("\n[2] インデックス作成中...")
            self.db.execute(create_index_query)
            self.db.commit()
            print("✅ ベクトル検索用インデックス作成完了")
            
            # テーブル確認
            check_query = f"""
            SELECT 
                table_name, 
                column_name, 
                data_type 
            FROM information_schema.columns 
            WHERE table_name = '{self.table_name}'
            ORDER BY ordinal_position;
            """
            
            print("\n[3] テーブル構造確認:")
            result = self.db.execute(check_query)
            if result:
                for row in result:
                    print(f"  - {row[1]}: {row[2]}")
            
            print("\n" + "=" * 50)
            print(f"✅ {self.table_name} 初期化完了")
            print(f"   モデルタイプ: {self.model_type}")
            print(f"   次元数: {self.embedding_dim}")
            print("=" * 50)
            
            return True
            
        except Exception as e:
            print(f"\n❌ エラー: {e}")
            return False
        finally:
            self.db.close()

    # 将来削除対象処理
    def drop_table(self):
        """テーブル削除（テスト用）"""
        if not self.db.connect():
            return False
        
        try:
            drop_query = f"DROP TABLE IF EXISTS {self.table_name} CASCADE;"
            self.db.execute(drop_query)
            self.db.commit()
            print(f"✅ {self.table_name} 削除完了")
            return True
        except Exception as e:
            print(f"❌ エラー: {e}")
            return False
        finally:
            self.db.close()
    
    # テーブル情報取得処理
    def get_table_info(self):
        """テーブル情報を取得"""
        if not self.db.connect():
            return None
        
        try:
            count_query = f"SELECT COUNT(*) FROM {self.table_name};"
            result = self.db.execute(count_query)
            count = result[0][0] if result else 0
            
            return {
                'model_type': self.model_type,
                'table_name': self.table_name,
                'embedding_dim': self.embedding_dim,
                'document_count': count
            }
        except Exception as e:
            print(f"❌ エラー: {e}")
            return None
        finally:
            self.db.close()

# テスト実行
if __name__ == "__main__":
    print("VectorStore テスト\n")
    
    # Google 768次元テスト
    print("=== Google 768次元テスト ===")
    vs_google = VectorStore(model_type='google-768')
    info = vs_google.get_table_info()
    if info:
        print(f"テーブル名: {info['table_name']}")
        print(f"次元数: {info['embedding_dim']}")
        print(f"文書数: {info['document_count']}")
    
    # Ollama 1024次元テスト
    print("\n=== Ollama 1024次元テスト ===")
    vs_ollama = VectorStore(model_type='ollama-1024')
    info = vs_ollama.get_table_info()
    if info:
        print(f"テーブル名: {info['table_name']}")
        print(f"次元数: {info['embedding_dim']}")
        print(f"文書数: {info['document_count']}")
    
    print("\n全テスト成功！")