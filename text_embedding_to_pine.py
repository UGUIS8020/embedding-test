import os
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

index_name = "text-search"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-west-2')
    )

index = pc.Index(index_name)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

def process_text(text_file_path):
    with open(text_file_path, 'r', encoding='utf-8') as f:
        return f.read()

def split_text(text):
    # テキストスプリッターの設定
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # チャンクサイズ（文字数）
        chunk_overlap=200,  # オーバーラップ（文字数）
        length_function=len,
        separators=["\n\n", "\n", "。", "、", " ", ""]  # 日本語テキスト用の区切り文字
    )
    
    # テキストを分割
    return text_splitter.split_text(text)

def main():
    data_directory = "./data/text/"
    
    for filename in os.listdir(data_directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(data_directory, filename)
            text = process_text(file_path)
            
            # テキストを分割
            chunks = split_text(text)
            
            # 各チャンクを処理
            for i, chunk in enumerate(chunks):
                vector = embedding_model.embed_query(chunk)
                
                metadata = {
                    "filename": filename,
                    "chunk_index": i,
                    "text": chunk,  # テキスト全体を保存
                    "data_type": "text_content",
                    "source": "e-sports"
                }
                
                # チャンク番号を含むユニークなIDを生成
                doc_id = f"text_{filename}_{i}"
                index.upsert([(doc_id, vector, metadata)])
                print(f"Processed chunk {i} of {filename}")

if __name__ == "__main__":
    main()