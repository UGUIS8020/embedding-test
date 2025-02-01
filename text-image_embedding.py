import os
from dataclasses import dataclass
from typing import List, Dict
import json
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from openai import OpenAI as OpenAIClient
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

index_name = "raiden"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-west-2')
    )

index = pc.Index(index_name)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

def read_text_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def find_related_images(text_file: str, image_directory: str) -> list:
    """テキストファイルに関連する画像ファイルを特定"""
    base_name = os.path.splitext(text_file)[0]
    parts = base_name.split('_')
    
    # Fig*の部分を抽出
    fig_identifiers = [part for part in parts if part.startswith('Fig')]
    
    images = []
    for file in os.listdir(image_directory):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            for fig_id in fig_identifiers:
                if fig_id.lower() in file.lower():
                    images.append(file)
                    break
    
    return sorted(images)

def generate_summary(text_content: str) -> str:
    """テキストの内容を50文字程度に要約"""
    try:
        client = OpenAIClient()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "あなたは医療文書の要約スペシャリストです。与えられたテキストを80文字程度の簡潔な要約にしてください。"},
                {"role": "user", "content": text_content}
            ],
            max_tokens=100,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"要約生成中にエラーが発生しました: {str(e)}")
        return "要約を生成できませんでした"

@dataclass
class TextImageGroup:
    text_file: str
    text_content: str
    summary: str  # 要約を追加
    related_images: List[str]
    metadata: Dict

def format_timestamp(timestamp):
    """UNIXタイムスタンプを年月日時分の形式に変換"""
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime('%Y-%m-%d %H:%M')

def process_directory(directory: str) -> List[TextImageGroup]:
    text_image_groups = []
    
    for file in os.listdir(directory):
        if file.endswith('.txt'):
            text_path = os.path.join(directory, file)
            text_content = read_text_file(text_path)
            
            summary = generate_summary(text_content)
            related_images = find_related_images(file, directory)
            
            # Noneを適切なデフォルト値に変更
            sequence_number = os.path.splitext(file)[0].split('_')[-1] if '_' in file else "0"  # Noneの代わりに"0"
            
            metadata = {
                "document_type": "technical_document",
                "parent_text": os.path.splitext(file)[0].split('_')[0],
                "summary": summary,
                "sequence_number": sequence_number,  # デフォルト値を使用                
                "creation_date": format_timestamp(os.path.getctime(text_path)),  # 例: "2025-02-01 15:45:23"
                "last_modified": format_timestamp(os.path.getmtime(text_path)),
                "file_size": os.path.getsize(text_path),
                "image_count": len(related_images)
            }
            
            group = TextImageGroup(
                text_file=file,
                text_content=text_content,
                summary=summary,
                related_images=related_images,
                metadata=metadata
            )
            text_image_groups.append(group)
            print(f"処理完了: {file} - 要約: {summary}")
    
    return text_image_groups

def create_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    """テキストを指定されたサイズでチャンク化"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "、", " ", ""]
    )
    return text_splitter.split_text(text)

def store_in_pinecone(text_image_groups: List[TextImageGroup], index) -> None:
    """チャンク化したテキストデータをベクトル化してPineconeに保存"""
    for group in text_image_groups:
        try:
            # テキストをチャンク化
            chunks = create_chunks(group.text_content)
            
            vectors = []
            # 各チャンクをベクトル化
            for i, chunk in enumerate(chunks):
                chunk_embedding = embedding_model.embed_query(chunk)
                
                vectors.append({
                    "id": f"{group.text_file}_chunk_{i}",
                    "values": chunk_embedding,
                    "metadata": {
                        "type": "content",
                        "text_file": group.text_file,
                        "chunk_index": i,
                        "chunk_text": chunk,
                        "summary": group.summary,
                        "related_images": group.related_images,
                        **group.metadata
                    }
                })
            
            # 要約のembedding
            summary_embedding = embedding_model.embed_query(group.summary)
            vectors.append({
                "id": f"{group.text_file}_summary",
                "values": summary_embedding,
                "metadata": {
                    "type": "summary",
                    "text_file": group.text_file,
                    "summary": group.summary,
                    "related_images": group.related_images,
                    **group.metadata
                }
            })
            
            # バッチでPineconeに保存
            index.upsert(vectors=vectors)
            print(f"Pineconeに保存完了: {group.text_file} (チャンク数: {len(chunks)})")
            
        except Exception as e:
            print(f"エラーが発生しました({group.text_file}): {str(e)}")

def save_to_json(groups: List[TextImageGroup], output_dir: str):
    """処理結果を個別のJSONファイルとして保存"""
    # 出力ディレクトリがない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 各グループを個別のJSONファイルとして保存
    for group in groups:  # text_image_groupsではなくgroupsを使用
        # ファイル名から.txtを除いてjsonを付ける
        base_name = os.path.splitext(group.text_file)[0]
        output_file = os.path.join(output_dir, f"{base_name}.json")
        
        data = {
            "text_file": group.text_file,
            "text_content": group.text_content,
            "summary": group.summary,
            "related_images": group.related_images,
            "metadata": group.metadata
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"JSONファイルを保存しました: {output_file}")

def store_in_pinecone(text_image_groups: List[TextImageGroup], index) -> None:
    """テキストデータをベクトル化してPineconeに保存"""
    for group in text_image_groups:
        try:
            # テキストコンテンツのembedding生成
            content_embedding = embedding_model.embed_query(group.text_content)
            
            # 要約のembedding生成
            summary_embedding = embedding_model.embed_query(group.summary)
            
            # Pineconeに保存するベクトルデータの準備
            vectors = [
                {
                    "id": f"{group.text_file}_content",
                    "values": content_embedding,
                    "metadata": {
                        "type": "content",
                        "text_file": group.text_file,
                        "text_content": group.text_content,
                        "summary": group.summary,
                        "related_images": group.related_images,
                        **group.metadata
                    }
                },
                {
                    "id": f"{group.text_file}_summary",
                    "values": summary_embedding,
                    "metadata": {
                        "type": "summary",
                        "text_file": group.text_file,
                        "summary": group.summary,
                        "related_images": group.related_images,
                        **group.metadata
                    }
                }
            ]
            
            # Pineconeにベクトルを保存
            index.upsert(vectors=vectors)
            print(f"Pineconeに保存完了: {group.text_file}")
            
        except Exception as e:
            print(f"エラーが発生しました({group.text_file}): {str(e)}")

def process_text(text_file_path):
    with open(text_file_path, 'r', encoding='utf-8') as f:
        return f.read()
    
def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", "。", "、", " ", ""]
    )
    return text_splitter.split_text(text)

def main():
    data_directory = "./data/test/"  # テキストと画像が同じフォルダに
    
    for filename in os.listdir(data_directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(data_directory, filename)
            text = process_text(file_path)
            chunks = split_text(text)
            
            # 同じディレクトリから関連画像を検索
            related_images = find_related_images(filename, data_directory)
            
            # メインテキストの要約を生成
            main_summary = generate_summary(text)
            
            # メインテキストのメタデータを保存
            main_metadata = {
                "filename": filename,
                "text": text,
                "summary": main_summary,
                "related_images": related_images,
                "data_type": "main_text",
                "source": "SHIBUYA Dental Laboratory",
                "image_count": len(related_images)
            }
            
            # メインテキストのベクトルを保存
            main_vector = embedding_model.embed_query(text)
            main_doc_id = f"main_{filename}"
            index.upsert([(main_doc_id, main_vector, main_metadata)])
            print(f"Processed main text: {filename}")
            
            # チャンクごとの処理
            for i, chunk in enumerate(chunks):
                chunk_summary = generate_summary(chunk)
                chunk_vector = embedding_model.embed_query(chunk)
                
                chunk_metadata = {
                    "filename": filename,
                    "chunk_index": i,
                    "text": chunk,
                    "summary": chunk_summary,
                    "related_images": related_images,
                    "data_type": "text_chunk",
                    "source": "SHIBUYA Dental Laboratory",
                    "parent_doc_id": main_doc_id
                }
                
                chunk_doc_id = f"chunk_{filename}_{i}"
                index.upsert([(chunk_doc_id, chunk_vector, chunk_metadata)])
                print(f"Processed chunk {i} of {filename}")

if __name__ == "__main__":
    main()