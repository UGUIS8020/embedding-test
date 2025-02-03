import os
from pinecone import Pinecone
from openai import OpenAI

# OpenAI clientの初期化
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Pineconeの初期化
pc = Pinecone(
    api_key=os.getenv('PINECONE_API_KEY')
)

# インデックスに接続
index = pc.Index("raiden")

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

print("\n=== 仮のベクトルでの検索結果 ===")
# 保存されているデータを確認（既存のコード）
results = index.query(
    vector=[0.1] * 1536,
    top_k=3,
    include_metadata=True
)

# 結果の表示（既存のコード）
print("\n保存されているデータの例:")
for match in results.matches:
    print(f"\nスコア: {match.score}")
    if hasattr(match, 'metadata'):
        print(f"メタデータ: {match.metadata}")

# 実際の検索機能を追加
print("\n=== 実際のクエリでの検索結果 ===")
test_queries = [
    "歯胚の形成過程について教えて",
    "歯根膜の構造と機能は？",
]

for query in test_queries:
    print(f"\n検索クエリ: {query}")
    query_embedding = get_embedding(query)
    
    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )
    
    print("\n上位3件の結果:")
    for match in results.matches:
        print(f"\nスコア: {match.score}")
        if hasattr(match, 'metadata'):
            if 'text' in match.metadata:
                print(f"テキスト: {match.metadata['text'][:200]}...")
            elif 'text_content' in match.metadata:
                print(f"テキスト: {match.metadata['text_content'][:200]}...")
            if 'summary' in match.metadata:
                print(f"要約: {match.metadata['summary']}")