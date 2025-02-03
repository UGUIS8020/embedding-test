import os
from pinecone import Pinecone
from openai import OpenAI
import time
from datetime import datetime

class MetadataPatternTester:
    def __init__(self):
        # Pineconeの初期化
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.index = self.pc.Index("raiden")
        
        # OpenAIの初期化
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def get_embedding(self, text):
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def run_test(self):
        # テストケースの定義
        test_cases = [
            {
                "query": "歯胚の形成過程について",
                "expected_keywords": ["歯胚", "形成", "エナメル器", "歯乳頭"],
                "expected_section": "歯と歯周組織の発生"
            },
            {
                "query": "歯根膜の構造と機能について",
                "expected_keywords": ["歯根膜", "シャーピー線維", "歯周組織"],
                "expected_section": "歯と歯周組織の解剖"
            }
        ]

        results = []
        
        for test in test_cases:
            print(f"\n=== テストクエリ: {test['query']} ===")
            
            # 検索実行
            start_time = time.time()
            query_embedding = self.get_embedding(test['query'])
            search_results = self.index.query(
                vector=query_embedding,
                top_k=3,
                include_metadata=True
            )
            query_time = time.time() - start_time

            # 結果の評価
            evaluation = self.evaluate_results(search_results, test, query_time)
            results.append(evaluation)
            
            # 結果の表示
            self.display_results(search_results, evaluation)

        return results

    def evaluate_results(self, search_results, test_case, query_time):
        evaluation = {
            "query": test_case["query"],
            "query_time": query_time,
            "matches": [],
            "keyword_matches": 0,
            "section_matches": 0,
            "avg_score": 0
        }

        scores = []
        for match in search_results.matches:
            score = match.score
            scores.append(score)
            
            # メタデータの評価
            metadata = match.metadata
            
            # キーワードマッチの確認
            found_keywords = 0
            if 'text' in metadata:
                for keyword in test_case["expected_keywords"]:
                    if keyword in metadata['text'].lower():
                        found_keywords += 1
            
            # セクションマッチの確認
            section_match = False
            if 'summary' in metadata:
                if test_case["expected_section"].lower() in metadata['summary'].lower():
                    section_match = True
            
            evaluation["matches"].append({
                "score": score,
                "keyword_matches": found_keywords,
                "section_match": section_match
            })

        evaluation["avg_score"] = sum(scores) / len(scores)
        evaluation["total_keyword_matches"] = sum(m["keyword_matches"] for m in evaluation["matches"])
        evaluation["total_section_matches"] = sum(1 for m in evaluation["matches"] if m["section_match"])

        return evaluation

    def display_results(self, search_results, evaluation):
        print(f"\n検索時間: {evaluation['query_time']:.3f}秒")
        print(f"平均スコア: {evaluation['avg_score']:.3f}")
        print(f"キーワードマッチ総数: {evaluation['total_keyword_matches']}")
        print(f"セクションマッチ数: {evaluation['total_section_matches']}")
        
        print("\n--- 検索結果の詳細 ---")
        for i, match in enumerate(search_results.matches):
            print(f"\n結果 {i+1}:")
            print(f"スコア: {match.score:.3f}")
            if 'summary' in match.metadata:
                print(f"要約: {match.metadata['summary']}")
            if 'text' in match.metadata:
                print(f"テキスト（先頭200文字）: {match.metadata['text'][:200]}...")

# テスターの実行
tester = MetadataPatternTester()
test_results = tester.run_test()