# chromaUUID,image無し
import os
import json
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv

# OpenAI API キーの設定
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# OpenAI Embeddings の初期化
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

def initialize_chroma(persist_directory):
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}  # UUIDを生成しないようにする
    )

def find_json_file(directory):
    for file in os.listdir(directory):
        if file.endswith('.json'):
            return os.path.join(directory, file)
    return None

def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def process_json_and_create_documents(json_data, directory):
    documents = []
    ids = []
    for item in json_data:
        page_number = item['page']
        for content in item['contents']:
            text_file_name = content['text_file']
            text_file_path = os.path.join(directory, text_file_name)
            
            if os.path.exists(text_file_path):
                text_content = read_text_file(text_file_path)
                
                metadata = {
                    "page": page_number,
                    "filename": text_file_name,
                    "chapter_title_en": json_data[0]['title']['en'] if 'title' in json_data[0] else "",
                    "chapter_title_ja": json_data[0]['title']['ja'] if 'title' in json_data[0] else ""
                }
                
                doc = Document(page_content=text_content, metadata=metadata)
                documents.append(doc)
                ids.append(f"page_{page_number}_{text_file_name}")  # カスタムIDの生成
            else:
                print(f"警告: ファイルが見つかりません: {text_file_path}")
    
    return documents, ids

def store_documents_in_chroma(chroma_db, documents, ids):
    chroma_db.add_documents(documents=documents, ids=ids)
    print(f"{len(documents)} ドキュメントがChromaデータベースに追加されました。")

def main():
    data_directory = "./data/渋谷歯科技工所"  # 指定されたディレクトリ
    persist_directory = "./my_chroma_app01"
    
    json_file_path = find_json_file(data_directory)
    if not json_file_path:
        print("JSONファイルが見つかりません。")
        return

    json_data = load_json_data(json_file_path)
    chroma_db = initialize_chroma(persist_directory)
    
    documents, ids = process_json_and_create_documents(json_data, data_directory)
    store_documents_in_chroma(chroma_db, documents, ids)

if __name__ == "__main__":
    main()