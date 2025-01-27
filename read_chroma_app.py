import os
import sqlite3
import json
import base64
from PIL import Image
import io

def analyze_chroma_folder(folder_path):
    print(f"Analyzing Chroma folder: {folder_path}")
    
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            if item.endswith('.sqlite3'):
                analyze_sqlite_file(item_path)
            else:
                analyze_unknown_file(item_path)

def analyze_sqlite_file(file_path):
    print(f"\nAnalyzing SQLite file: {os.path.basename(file_path)}")
    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()
    
    # Get table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables:")
    for table in tables:
        print(f"- {table[0]}")
        analyze_table(cursor, table[0])

    conn.close()

def analyze_table(cursor, table_name):
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    print(f"  Columns in {table_name}:")
    for column in columns:
        print(f"    - {column[1]} ({column[2]})")
    
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    print(f"  Total rows: {count}")
    
    if table_name == 'embeddings':
        analyze_embeddings(cursor)

def analyze_embeddings(cursor):
    cursor.execute("SELECT embedding_id, created_at FROM embeddings LIMIT 5")
    samples = cursor.fetchall()
    print("\nSample embeddings:")
    for i, (document, metadata) in enumerate(samples, 1):
        print(f"\nSample {i}:")
        print(f"  Document: {document[:100]}..." if len(document) > 100 else document)
        try:
            metadata_dict = json.loads(metadata)
            print("  Metadata:")
            for key, value in metadata_dict.items():
                print(f"    {key}: {value}")
            if 'filename' in metadata_dict:
                _, ext = os.path.splitext(metadata_dict['filename'])
                if ext.lower() in ['.png', '.jpg', '.jpeg', '.gif']:
                    print("  This embedding represents an image file.")
                elif ext.lower() in ['.json']:
                    print("  This embedding represents a JSON file.")
                elif ext.lower() in ['.txt']:
                    print("  This embedding represents a text file.")
        except json.JSONDecodeError:
            print("  Error: Could not parse metadata as JSON")

def analyze_unknown_file(file_path):
    print(f"\nAnalyzing unknown file: {os.path.basename(file_path)}")
    with open(file_path, 'rb') as f:
        content = f.read()
    
    # Try to parse as JSON
    try:
        json_data = json.loads(content)
        print("File content is JSON:")
        print(json.dumps(json_data, indent=2)[:500] + "..." if len(json.dumps(json_data)) > 500 else json.dumps(json_data, indent=2))
        return
    except json.JSONDecodeError:
        pass
    
    # Try to open as image
    try:
        image = Image.open(io.BytesIO(content))
        print(f"File is an image: {image.format}, {image.size}x{image.mode}")
        return
    except IOError:
        pass
    
    # If not JSON or image, show as text or binary
    try:
        text_content = content.decode('utf-8')
        print("File content (first 500 characters):")
        print(text_content[:500] + "..." if len(text_content) > 500 else text_content)
    except UnicodeDecodeError:
        print("File contains binary data")
        print(f"File size: {len(content)} bytes")

def main():
    chroma_folder = "./chroma_db"
    analyze_chroma_folder(chroma_folder)

if __name__ == "__main__":
    main()