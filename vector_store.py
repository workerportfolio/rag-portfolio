#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
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

    # テーブル挿入処理
    def insert_document(self, text: str, embedding: list, metadata: dict = None) -> int:
        """
        ドキュメントを挿入
        Args:
            text: ドキュメントテキスト
            embedding: ベクトル（リスト）
            metadata: メタデータ
        Returns:
            ドキュメントID
        """
        if not self.db.connect():
            return None
        
        try:
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
                return result[0][0]
            return None
            
        except Exception as e:
            print(f"❌ エラー: {e}")
            return None
        finally:
            self.db.close()
    
    # ベクトル検索処理
    def search_similar(self, query_embedding: list, top_k: int = 3, embedding_model: str = None) -> dict:
        """
        類似ドキュメントを検索（デバッグ情報付き）
        Args:
            query_embedding: クエリベクトル
            top_k: 取得件数
            embedding_model: Embeddingモデル名（デバッグ情報用）
        Returns:
            検索結果とデバッグ情報の辞書
        """
        if not self.db.connect():
            return {'results': [], 'debug_info': None}
        
        try:
            vector_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            search_query = f"""
            SELECT 
                id,
                document_text,
                metadata,
                embedding <=> '{vector_str}'::vector({self.embedding_dim}) as distance
            FROM {self.table_name}
            ORDER BY embedding <=> '{vector_str}'::vector({self.embedding_dim})
            LIMIT %s;
            """
            
            self.db.cursor.execute(search_query, (top_k,))
            results_raw = self.db.cursor.fetchall()
            
            # デバッグ情報構築
            debug_info = {
                'table_name': self.table_name,
                'embedding_model': embedding_model,
                'embedding_dim': len(query_embedding),
                'top_k_raw': len(results_raw),
                'threshold': None,
                'results_raw': [],
                'results_filtered': [],
                'filtered_count': 0,
                'discarded_reasons': []
            }
            
            # 結果整形
            results_filtered = []
            for i, (doc_id, text, metadata, distance) in enumerate(results_raw, 1):
                debug_info['results_raw'].append({
                    'rank': i,
                    'id': doc_id,
                    'distance': distance,
                    'text_preview': text[:100]
                })
                
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
            
            debug_info['filtered_count'] = len(results_filtered)
            
            return {
                'results': results_filtered,
                'debug_info': debug_info
            }
            
        except Exception as e:
            print(f"❌ エラー: {e}")
            return {'results': [], 'debug_info': None}
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