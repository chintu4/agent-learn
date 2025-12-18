# Complete AI Agent + Web Scraper Integration
# Combines your TinyLlama agent with the news scraper
# pip install llama-cpp-python requests beautifulsoup4 lxml

from llama_cpp import Llama
import requests
from bs4 import BeautifulSoup
import json
import csv
from datetime import datetime
import time
import re

# Load your TinyLlama model (agent brain)
model_path = "C:/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=4,
    verbose=False
)

class NewsScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def scrape_google_news(self, max_articles=10):
        url = "https://news.google.com"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            headline_elements = soup.find_all('a', class_='DY5T1d')[:max_articles]
            
            articles = []
            for elem in headline_elements:
                title = elem.get_text().strip()
                href = elem.get('href') if hasattr(elem, 'get') else ''
                if not href:
                    link = ''
                else:
                    if isinstance(href, str):
                        href_str = href
                    elif isinstance(href, (list, tuple)):
                        href_str = href[0] if href else ''
                    else:
                        href_str = str(href)
                    if href_str.startswith('/'):
                        link = f"https://news.google.com{href_str}"
                    else:
                        link = href_str
                articles.append({'title': title, 'link': link, 'source': 'Google News'})
            return articles
        except:
            return []

# TOOL: Web scraping function for agent
def scrape_news_tool(source="google", max_articles=10):
    """Agent tool: Scrape news headlines"""
    scraper = NewsScraper()
    if source.lower() == "google":
        return scraper.scrape_google_news(max_articles)
    return []

# AGENT: Main agentic workflow
def agent(prompt, tools_available=True, max_tokens=512):
    system_prompt = """You are an AI agent that can scrape web news and analyze data.
    Use tools when needed. Think step-by-step. Respond in JSON: {"thought": "...", "action": "...", "result": "..."}"""
    
    full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"
    
    response = llm(
        full_prompt,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        stop=["<|user|>", "<|system|>"]
    )
    
    agent_output = response['choices'][0]['text'].strip()
    
    # Parse agent JSON response
    try:
        parsed = json.loads(agent_output)
        thought = parsed.get('thought', '')
        action = parsed.get('action', '').lower()
        
        # Execute tools based on agent decision
        if tools_available and action == 'scrape_news':
            news_data = scrape_news_tool()
            print("📰 AGENT TOOL EXECUTED: Scraped news headlines")
            print(f"Found {len(news_data)} articles")
            return f"AGENT RESULT: Scraped {len(news_data)} headlines. Thought: {thought}"
        
        return agent_output
    except:
        return agent_output

# MAIN: Agent-driven news workflow
def main():
    print("🤖 TinyLlama AI Agent + Web Scraper Active")
    print("=" * 50)
    
    # Agent task 1: Plan scraping strategy
    task1 = agent("Plan how to get top 10 news headlines from Google News")
    print("🧠 AGENT PLAN:", task1)
    
    # Agent task 2: Execute scraping with tools
    task2 = agent("Scrape latest news headlines now", tools_available=True)
    print("\n🔍 AGENT EXECUTION:", task2)
    
    # Agent task 3: Analyze results
    news_data = scrape_news_tool(max_articles=10)
    analysis_prompt = f"Analyze these headlines: {json.dumps(news_data[:3])} What are the main topics?"
    task3 = agent(analysis_prompt)
    print("\n📊 AGENT ANALYSIS:", task3)
    
    # Save results
    if news_data:
        filename = f"agent_news_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        with open(filename, 'w') as f:
            json.dump(news_data, f, indent=2)
        print(f"\n💾 Saved {len(news_data)} articles to {filename}")

if __name__ == "__main__":
    main()
