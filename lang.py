import requests
from bs4 import BeautifulSoup
import os
import json
from github import Github
from dotenv import load_dotenv
from urllib.parse import urljoin

# 環境変数の読み込み
load_dotenv()

# 出力ディレクトリの設定
base_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lang')
docs_output_dir = os.path.join(base_output_dir, 'docs')
github_output_dir = os.path.join(base_output_dir, 'github')
notebooks_output_dir = os.path.join(base_output_dir, 'notebooks')

os.makedirs(docs_output_dir, exist_ok=True)
os.makedirs(github_output_dir, exist_ok=True)
os.makedirs(notebooks_output_dir, exist_ok=True)

print(f"Output directories: \n- Docs: {docs_output_dir}\n- GitHub: {github_output_dir}\n- Notebooks: {notebooks_output_dir}")

# GitHubトークンの取得
github_token = os.getenv('GITHUB_TOKEN')
if not github_token:
    print("Warning: GITHUB_TOKEN not found in environment variables. Using unauthenticated requests.")

def get_github_client():
    if github_token:
        return Github(github_token)
    else:
        return Github()

def scrape_langchain_docs():
    base_url = "https://python.langchain.com/docs/get_started/introduction"
    visited_urls = set()
    to_visit = [base_url]
    scraped_data = []

    def extract_content(soup):
        main_content = soup.find('main')
        if main_content:
            # Remove navigation elements
            for nav in main_content.find_all(['nav', 'header', 'footer']):
                nav.decompose()
            return main_content.get_text(strip=True)
        return ""

    while to_visit:
        url = to_visit.pop(0)
        if url in visited_urls:
            continue

        try:
            print(f"Scraping: {url}")
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            content = extract_content(soup)
            scraped_data.append({"url": url, "content": content})
            visited_urls.add(url)

            # Find links to other pages within the same domain
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(url, href)
                if full_url.startswith(base_url) and full_url not in visited_urls:
                    to_visit.append(full_url)

        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")

    print(f"Scraped {len(scraped_data)} pages")
    return scraped_data

def get_github_content():
    try:
        print("Connecting to GitHub...")
        g = get_github_client()
        repo = g.get_repo("langchain-ai/langchain")
        contents = repo.get_contents("")
        data = []
        for content_file in contents:
            if content_file.type == "file" and content_file.name.endswith(('.md', '.py')):
                try:
                    file_content = content_file.decoded_content.decode('utf-8')
                    data.append({
                        'name': content_file.name,
                        'path': content_file.path,
                        'content': file_content
                    })
                    print(f"Fetched: {content_file.path}")
                except Exception as e:
                    print(f"Error fetching content for {content_file.path}: {str(e)}")
        print(f"Fetched {len(data)} files from GitHub")
        return data
    except Exception as e:
        print(f"Error fetching GitHub content: {str(e)}")
        return []

def download_jupyter_notebooks():
    def explore_directory(repo, path):
        contents = repo.get_contents(path)
        for content in contents:
            if content.type == "dir":
                yield from explore_directory(repo, content.path)
            elif content.name.endswith('.ipynb'):
                yield content

    try:
        print("Connecting to GitHub for Jupyter Notebooks...")
        g = get_github_client()
        repo = g.get_repo("langchain-ai/langchain")
        
        notebooks = list(explore_directory(repo, ""))
        print(f"Found {len(notebooks)} Jupyter Notebooks")
        
        for notebook in notebooks:
            try:
                print(f"Downloading: {notebook.path}")
                notebook_content = notebook.decoded_content
                notebook_path = os.path.join(notebooks_output_dir, notebook.name)
                with open(notebook_path, 'wb') as f:
                    f.write(notebook_content)
                print(f"Successfully downloaded: {notebook.name}")
            except Exception as e:
                print(f"Error downloading {notebook.name}: {str(e)}")
        
        print(f"Downloaded {len(notebooks)} notebooks")
    except Exception as e:
        print(f"Error exploring GitHub repository: {str(e)}")

def main():
    try:
        # LangChain公式ドキュメントのスクレイピング
        print("Scraping LangChain official docs...")
        docs_content = scrape_langchain_docs()
        if docs_content:
            output_file = os.path.join(docs_output_dir, 'langchain_docs.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(docs_content, f, ensure_ascii=False, indent=2)
            print(f"Official docs scraped and saved to: {output_file}")
        else:
            print("No content scraped from official docs.")

        # GitHubコンテンツの取得
        print("Fetching GitHub content...")
        github_content = get_github_content()
        if github_content:
            output_file = os.path.join(github_output_dir, 'langchain_github_content.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(github_content, f, ensure_ascii=False, indent=2)
            print(f"GitHub content fetched and saved to: {output_file}")
        else:
            print("No content fetched from GitHub.")

        # Jupyter Notebooksのダウンロード
        print("Downloading Jupyter Notebooks...")
        download_jupyter_notebooks()
        print("Jupyter Notebooks download process completed.")

        # 最終結果の表示
        print("\nFinal Results:")
        print(f"Files in output directory '{base_output_dir}':")
        for folder in [docs_output_dir, github_output_dir, notebooks_output_dir]:
            print(f"\nContents of {folder}:")
            for file in os.listdir(folder):
                print(f"- {file}")
        
        # ファイルの内容サマリーを表示
        print("\nFile Content Summary:")
        for folder in [docs_output_dir, github_output_dir, notebooks_output_dir]:
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if file.endswith('.json'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            print(f"{file}: {len(data)} items")
                        else:
                            print(f"{file}: JSON object")
                elif file.endswith('.ipynb'):
                    print(f"{file}: Jupyter Notebook file")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()