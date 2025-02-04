import os
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv

"""pineconeに保存されたテキストと画像が紐づけされたデータの確認"""
"""試しに検索ワードで確認してみる"""

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

def search_with_keyword(query_text):
    """キーワードで検索を行う"""
    index = initialize_pinecone()
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # クエリベクトルの生成
    query_vector = embedding_model.embed_query(query_text)
    
    # データの取得
    results = index.query(
        vector=query_vector,
        top_k=1,  # 最も関連性の高い結果を1件取得
        include_metadata=True
    )
    
    print(f"\n=== 検索キーワード: '{query_text}' ===")
    for match in results.matches:
        print(f"\n類似度スコア: {match.score:.4f}")
        if match.metadata:
            print("\nテキスト:")
            print(match.metadata.get('text', ''))
            print("\n関連画像:")
            for img in match.metadata.get('related_images', []):
                print(f"- {img}")

# テスト用の検索キーワード
keywords = [
    "歯胚",
    "エナメル器",
    "歯と歯周組織の発生",
    "象牙質形成"
]

# 各キーワードで検索を実行
for keyword in keywords:
    search_with_keyword(keyword)

if __name__ == "__main__":
    check_stored_data()