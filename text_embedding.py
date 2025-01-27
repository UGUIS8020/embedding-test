import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from dotenv import load_dotenv

# OpenAI API キー
load_dotenv()
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

def initialize_chroma(persist_directory):
   if not os.path.exists(persist_directory):
       os.makedirs(persist_directory)
   return Chroma(
       persist_directory=persist_directory,
       embedding_function=embedding_model,
       collection_metadata={"hnsw:space": "cosine"}
   )

def process_text(text_file_path):
   with open(text_file_path, 'r', encoding='utf-8') as f:
       return f.read()

def create_documents(text_content, filename):
   metadata = {"filename": filename}
   return Document(page_content=text_content, metadata=metadata)

def store_in_chroma(chroma_db, document, id):
   chroma_db.add_documents(documents=[document], ids=[id])

def main():
   data_directory = "./data/text/"
   persist_directory = "./chroma_text_db"
   
   chroma_db = initialize_chroma(persist_directory)
   
   for filename in os.listdir(data_directory):
       if filename.endswith('.txt'):
           file_path = os.path.join(data_directory, filename)
           text_content = process_text(file_path)
           document = create_documents(text_content, filename)
           store_in_chroma(chroma_db, document, filename)
           print(f"Processed: {filename}")

if __name__ == "__main__":
   main()