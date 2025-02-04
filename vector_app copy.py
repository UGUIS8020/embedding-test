import os
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
"""pineconeに保存されたテキストと画像が紐づけされたデータの確認"""

def initialize_pinecone():
    """Pineconeの初期化"""
    load_dotenv()
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    return pc.Index("text-search")

def check_stored_data():
    """保存されたデータの確認"""
    # Pineconeの初期化
    index = initialize_pinecone()
    
    # クエリベクトルの生成（テスト用）
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    query = "テスト用クエリ"  # 任意のクエリ
    query_vector = embedding_model.embed_query(query)
    
    # データの取得
    results = index.query(
        vector=query_vector,
        top_k=10,  # 上位10件を取得
        include_metadata=True
    )
    
    print("\n=== 保存されたデータの確認 ===")
    for i, match in enumerate(results.matches, 1):
        print(f"\n--- 結果 {i} ---")
        print(f"ID: {match.id}")
        print(f"類似度スコア: {match.score:.4f}")
        
        # メタデータの表示
        if match.metadata:
            print("\nメタデータ:")
            print(f"ファイル名: {match.metadata.get('filename', 'なし')}")
            print(f"チャンク番号: {match.metadata.get('chunk_index', 'なし')}")
            
            # 関連画像の確認
            images = match.metadata.get('related_images', [])
            if images:
                print(f"\n関連画像 ({len(images)}件):")
                for img in images:
                    print(f"- {img}")
            
            # テキストの一部を表示
            text = match.metadata.get('text', '')
            if text:
                print("\nテキスト（最初の100文字）:")
                print(text[:100] + "...")
        
        print("-" * 50)

if __name__ == "__main__":
    check_stored_data()