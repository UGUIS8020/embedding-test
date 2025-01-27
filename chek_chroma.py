# chromaデータ確認

from chromadb import Client
from chromadb.config import Settings

# Chromaクライアントを初期化（パスを正確に指定）
client = Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma03_db"
))

# 利用可能なコレクションを取得
collections = client.list_collections()

if not collections:
    print("No collections found in the database.")
else:
    for collection in collections:
        print(f"\nChecking collection: {collection.name}")
        
        # コレクション内のすべてのアイテムを取得
        results = collection.get()
        
        if not results['ids']:
            print("  This collection is empty.")
        else:
            for i, (id, embedding, metadata) in enumerate(zip(results['ids'], results['embeddings'], results['metadatas'])):
                print(f"  Item {i + 1}:")
                print(f"    ID: {id}")
                print(f"    Embedding: {embedding[:5]}... (truncated)")
                print(f"    Metadata: {metadata}")
                print()

        # コレクションの追加情報を表示
        print(f"  Total items in collection: {collection.count()}")