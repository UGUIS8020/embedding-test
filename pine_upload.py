import os
import numpy as np
import json
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# Pineconeの初期化
pinecone_api_key = os.getenv('PINECONE_API_KEY')
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

pc = Pinecone(api_key=pinecone_api_key)
index_name = "shibuya"

# 既存のインデックスを削除（もし存在する場合）
if index_name in pc.list_indexes().names():
    print(f"Deleting existing index: {index_name}")
    pc.delete_index(index_name)

# 3072次元で新しいインデックスを作成
print(f"Creating new index with dimension 3072: {index_name}")
pc.create_index(
    name=index_name,
    dimension=3072,  # 次元数を3072に変更
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

index = pc.Index(index_name)

def load_embeddings(embeddings_folder):
    """embeddingsフォルダーからデータを読み込む"""
    embeddings_path = os.path.join(embeddings_folder, "combined_embeddings.npy")
    embeddings_dict = np.load(embeddings_path, allow_pickle=True).item()
    
    metadata_path = os.path.join(embeddings_folder, "content_metadata.json")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_dict = json.load(f)
    
    return embeddings_dict, metadata_dict

def upload_to_pinecone(embeddings_dict, metadata_dict, index):
    """PineconeにEmbeddingデータをアップロード"""
    batch_size = 100
    vectors = []
    
    for content_id, embeddings in embeddings_dict.items():
        print(f"\nProcessing content_id: {content_id}")
        
        # embeddingの存在確認とデバッグ出力
        print("Available embeddings:")
        for key, emb in embeddings.items():
            if isinstance(emb, np.ndarray):
                print(f"- {key}: shape={emb.shape}")

        try:
            # 全てのembeddingを結合
            combined_embedding = np.concatenate([
                embeddings.get('text', np.zeros(1536)),
                embeddings.get('image', np.zeros(512)),
                embeddings.get('metadata', np.zeros(1024))
            ])

            # メタデータを適切な形式に変換
            metadata_content = metadata_dict.get(content_id, {})
            formatted_metadata = {
                'content_id': str(content_id),  # 文字列に変換
                'text': str(metadata_content.get('text', '')),  # 文字列に変換
                'image_path': str(metadata_content.get('image_path', '')),  # 文字列に変換
                # 複雑なメタデータはJSON文字列として保存
                'metadata': json.dumps(metadata_content.get('metadata', {}))
            }

            vector = {
                'id': str(content_id),  # IDも文字列に変換
                'values': combined_embedding.tolist(),
                'metadata': formatted_metadata
            }
            vectors.append(vector)

            if len(vectors) >= batch_size:
                try:
                    index.upsert(vectors=vectors)
                    print(f"Uploaded {len(vectors)} vectors")
                    vectors = []
                except Exception as e:
                    print(f"Error uploading batch: {str(e)}")
                    raise e

        except Exception as e:
            print(f"Error processing {content_id}: {str(e)}")
            continue

    # 残りのベクトルをアップロード
    if vectors:
        try:
            index.upsert(vectors=vectors)
            print(f"Uploaded final {len(vectors)} vectors")
        except Exception as e:
            print(f"Error uploading final batch: {str(e)}")
            raise e

def main():
    embeddings_folder = "./embeddings"
    
    if not os.path.exists(embeddings_folder):
        raise ValueError(f"Embeddings folder not found: {embeddings_folder}")
    
    print("Loading embeddings and metadata...")
    embeddings_dict, metadata_dict = load_embeddings(embeddings_folder)
    
    print("Uploading to Pinecone...")
    upload_to_pinecone(embeddings_dict, metadata_dict, index)
    
    print("Upload complete!")

if __name__ == "__main__":
    main()