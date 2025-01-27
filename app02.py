import os
import json
from PIL import Image
import io
import base64
from openai import OpenAI
# from langchain_community.vectorstores import Chroma
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import CLIPProcessor, CLIPModel

# OpenAI API キーの設定
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

# OpenAI Embeddings の初期化
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Chroma の初期化
persist_directory = "./chroma_db"
chroma_db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

def process_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    
    # テキストまたはJSONファイルの処理
    if file_extension in ['.json', '.txt']:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # if file_extension == '.json':
        #     content = json.dumps(json.loads(content))  # JSON を整形された文字列に変換

        if file_extension == '.json':
            content = json.loads(content)  # 辞書形式でJSONを扱う
        
        # テキストのエンベディングを生成
        embedding = embedding_model.embed_query(content)
        return content, embedding
    
    # 画像ファイルの処理
    elif file_extension in ['.png', '.jpg', '.jpeg']:
        with open(file_path, 'rb') as image_file:
            image_data = image_file.read()
        
        # Base64エンコードされた画像データを準備
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # OpenAI APIを使用して画像の説明を生成
        try:
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",  # 更新されたモデル名
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "この画像を簡潔に説明してください。"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            description = response.choices[0].message.content
        except Exception as e:
            print(f"Error processing image {file_path}: {str(e)}")
            return None, None
        
        # 画像の説明文のエンベディングを生成
        embedding = embedding_model.embed_query(description)
        
        return description, embedding
    
    else:
        return None, None

def main():
    data_directory = "./data/lang"
    
    if not os.path.exists(data_directory):
        print(f"ディレクトリが存在しません: {data_directory}")
        return
    
    for filename in os.listdir(data_directory):
        file_path = os.path.join(data_directory, filename)
        content, embedding = process_file(file_path)
        
        if content is not None and embedding is not None:
            _, file_extension = os.path.splitext(filename)
            metadata = {
                "filename": filename,
                "type": get_file_type(file_extension),
                "file_extension": file_extension[1:]  # 先頭の'.'を除去
            }
            
            # Documentオブジェクトを作成
            doc = Document(page_content=content, metadata=metadata)
            
            try:
                # add_documents メソッドを使用
                chroma_db.add_documents([doc], ids=[filename])
                print(f"ファイルを処理しました: {filename}")
            except Exception as e:
                print(f"Error adding document {filename} to Chroma: {str(e)}")

    print("処理が完了しました。ベクトルストアに保存されました。")

def get_file_type(file_extension):
    if file_extension.lower() in ['.png', '.jpg', '.jpeg']:
        return "image"
    elif file_extension.lower() == '.json':
        return "json"
    else:
        return "text"

if __name__ == "__main__":
    main()