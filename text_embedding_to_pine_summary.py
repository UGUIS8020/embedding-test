import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

"""pineconeにテキストデータと画像データを紐づけして保存する"""

source_name = os.getenv('SOURCE_NAME', 'autologous tooth transplantation')

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

# カスタムプロンプトテンプレートの作成
summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
あなたは厳密に80文字以内で要約を作成するエキスパートです。
以下のルールを厳守してください：
1. 要約は必ず80文字以内に収めること
2. 重要な情報を優先的に含めること
3. 生成した要約の文字数を数えて、80文字を超える場合は短く修正すること

テキスト:
{text}

要約:"""
)

def generate_summary(text):
    try:
        # 新しい方法でチェーンを作成
        chain = summary_prompt | llm
        
        # invoke メソッドを使用して要約を生成
        summary = chain.invoke({"text": text})
        return summary.content.strip()
    except Exception as e:
        print(f"Error generating summary: {e}")
        return ""
    
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
    data_directory = "./data/text/"
    
    for filename in os.listdir(data_directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(data_directory, filename)
            text = process_text(file_path)
            chunks = split_text(text)
            
            for i, chunk in enumerate(chunks):
                summary = generate_summary(chunk)
                vector = embedding_model.embed_query(chunk)
                
                metadata = {
                    "filename": filename,
                    "chunk_index": i,
                    "text": chunk,
                    "summary": summary,
                    "data_type": "text_content",                    
                }
                
                doc_id = f"text_{filename}_{i}"
                index.upsert([(doc_id, vector, metadata)])
                print(f"Processed chunk {i} of {filename}")

if __name__ == "__main__":
    main()