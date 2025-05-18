import os
import requests
from dotenv import load_dotenv
from crawl4ai import *
import openai
import asyncio
from sentence_transformers import SentenceTransformer, util
import requests

# Initialize model (you can use 'all-MiniLM-L6-v2' or better if needed)
model = SentenceTransformer('all-MiniLM-L6-v2')

def choose_best_repo_by_prompt(prompt, items):
    descriptions = [item.get('description', '') or '' for item in items]
    texts = [prompt] + descriptions

    # Step 1: Get embeddings
    embeddings = model.encode(texts, convert_to_tensor=True)

    # Step 2: Compute similarities (prompt vs each description)
    similarities = util.cos_sim(embeddings[0], embeddings[1:])[0]

    # Step 3: Find best match
    best_idx = similarities.argmax().item()
    best_repo = items[best_idx]
    return best_repo, similarities[best_idx].item()

# GitHub API call (example)
response = requests.get("https://api.github.com/search/repositories?q=sas+language:python")
data = response.json()
items = data.get("items", [])

# Step 4: Take user input
user_prompt = "Provides simple scripts about options to deploy monitoring, alerts, and log aggregation for Viya 4 running on Kubernetes"

# Step 5: Find the most relevant repository
best_repo, score = choose_best_repo_by_prompt(user_prompt, items)

# Output result
print(f"Best match (score={score:.2f}): {best_repo['full_name']}")
print(f"Description: {best_repo['description']}")
print(f"URL: {best_repo['html_url']}")

# Now you can pass best_repo['html_url'] to crawl4ai

load_dotenv()


# Warm up the crawler (load necessary models)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}


def search_sas_repos(query):
    url = f"https://api.github.com/search/repositories?q=org:sassoazureftware"
    response = requests.get(url, headers=HEADERS)
    repos = response.json().get("items", [])
    return [repo['html_url'] for repo in repos]

async def crawl_repository(repo_url):
    print("Crawling repo: {repo_url}")
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url=repo_url,
        )
        return result.markdown

def get_answer_from_context(query, context):
    prompt = f"""Answer the following question using the provided GitHub repository context:

Context:
{context[:3000]}  # Truncate if too long

Question: {query}
Answer:"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

def ai_agent_pipeline(user_query):
    print("Searching relevant SAS repos...")
    repo_urls = search_sas_repos(user_query)

    if not repo_urls:
        return "No relevant repositories found."

    print(f"Crawling repo: {repo_urls[0]}")
    crawled_data = asyncio.run(crawl_repository(repo_urls[0]))
    print(crawled_data)
    print("Generating answer...")
    return get_answer_from_context(user_query, crawled_data)

# Example usage
if __name__ == "__main__":
    query = input("Ask your question: ")
    answer = ai_agent_pipeline(query)
    print("\nAnswer:\n", answer)
