# 比較的短い文章 jsonファイルとテキストファイルを処理して、Chromaデータベースに追加するスクリプトを作成しました

import json
import os
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import chromadb

# OpenAI API キーの設定
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

def process_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    documents = []
    current_section = ""
    current_content = []

    for line in lines:
        line = line.strip()
        if line and not line.startswith(" "):  # 新しいセクションの開始
            if current_section:  # 前のセクションを保存
                doc = Document(
                    page_content=f"{current_section}\n{''.join(current_content)}",
                    metadata={"source": file_path, "type": "text", "section": current_section}
                )
                documents.append(doc)
            current_section = line
            current_content = []
        elif line:
            current_content.append(line + "\n")
    
    # 最後のセクションを保存
    if current_section:
        doc = Document(
            page_content=f"{current_section}\n{''.join(current_content)}",
            metadata={"source": file_path, "type": "text", "section": current_section}
        )
        documents.append(doc)
    
    return documents

def process_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    for key, value in data.items():
        content = json.dumps(value, ensure_ascii=False, indent=2) if isinstance(value, (dict, list)) else str(value)
        doc = Document(
            page_content=f"{key}:\n{content}",
            metadata={"source": file_path, "type": "json", "key": key}
        )
        documents.append(doc)
    
    return documents

def get_base_filename(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

from chromadb.config import Settings

# ... (他のインポートと既存のコード)

def main():
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # ファイルパスを定義
    text_file = "data/shibuya.txt"
    json_file = "data/shibuya.json"
    
    # ベースファイル名を取得
    base_filename = get_base_filename(text_file)  # または json_file を使用
    
    # 永続化ディレクトリを設定
    persist_directory = f"./chroma03_db_{base_filename}"
    
    # Chromaクライアントを初期化
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    
    # Langchainの Chroma ベクトルストアを初期化
    chroma_db = Chroma(
        client=chroma_client,
        embedding_function=embedding_model,
        collection_name=f"{base_filename}_collection"
    )
    
    # テキストファイルの処理
    text_docs = process_text_file(text_file)
    chroma_db.add_documents(text_docs)
    print(f"テキストドキュメント {len(text_docs)} 件を追加しました。")
    
    # JSONファイルの処理
    json_docs = process_json_file(json_file)
    chroma_db.add_documents(json_docs)
    print(f"JSONドキュメント {len(json_docs)} 件を追加しました。")
    
    print(f"処理が完了しました。Chromaデータベースが {persist_directory} に保存されました。")

if __name__ == "__main__":
    main()