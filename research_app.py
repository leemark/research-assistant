import streamlit as st
import requests
import google.generativeai as genai
from typing import List, Dict
import os
from datetime import datetime
from bs4 import BeautifulSoup
import concurrent.futures
import time
from urllib.parse import urlparse

# Configure page settings
st.set_page_config(
    page_title="Deep Research Assistant",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'analysis' not in st.session_state:
    st.session_state.analysis = ""

# Configure API keys
BRAVE_API_KEY = st.secrets["BRAVE_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Updated Gemini model configuration
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 65536,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-thinking-exp-01-21",
    generation_config=generation_config,
)

def brave_search(query: str, num_results: int = 10, max_retries: int = 3) -> List[Dict]:
    """
    Perform a search using Brave Search API with retry and backoff in case of rate limiting.
    """
    headers = {
        "X-Subscription-Token": BRAVE_API_KEY,
        "Accept": "application/json",
    }
    
    url = "https://api.search.brave.com/res/v1/web/search"
    params = {
        "q": query,
        "count": num_results
    }
    
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            results = response.json().get('web', {}).get('results', [])
            return results
        elif response.status_code == 429:
            st.warning("Rate limited by Brave Search API (429). Retrying...")
            time.sleep(2 * (attempt + 1))
        else:
            st.error(f"Search API Error: {response.status_code}")
            break
    return []

def scrape_webpage(url: str) -> str:
    """
    Scrape content from a webpage with error handling
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text content
        text = soup.get_text(separator=' ', strip=True)
        
        # Basic text cleaning
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Limit text length to avoid token limits
        return text[:10000]  # Adjust limit as needed
        
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"

def scrape_urls_parallel(urls: List[str]) -> Dict[str, str]:
    """
    Scrape multiple URLs in parallel
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(scrape_webpage, url): url for url in urls}
        results = {}
        
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                results[url] = future.result()
            except Exception as e:
                results[url] = f"Error: {str(e)}"
    
    return results

def analyze_with_gemini(query: str, search_results: List[Dict]) -> str:
    """
    Analyze search results using Gemini 2.0 Flash Thinking
    """
    # Scrape webpage contents from provided sources
    urls = [result['url'] for result in search_results]
    webpage_contents = scrape_urls_parallel(urls)
    
    current_time = datetime.now().isoformat()
    prompt = f"""
    You are an expert researcher. Today is {current_time}. Follow these instructions when responding:
    - You may be asked to research subjects that are post your knowledge cutoff; assume the user is right when new information is presented.
    - The user is a highly experienced analyst, so be as detailed and accurate as possible.
    - Be highly organized, proactive, and anticipate further needs.
    - Suggest solutions or insights that might not have been considered.
    - Provide detailed explanations and analysis.
    - Value good arguments over mere authority; however, reference the provided sources explicitly using inline citations (e.g., [Source 1]).
    - Consider new technologies and contrarian ideas, and clearly flag any high levels of speculation.
    
    Research Query: {query}
    
    Below are the sources and their content summaries:
    {'-' * 50}
    """
    
    for idx, result in enumerate(search_results, 1):
        url = result['url']
        content = webpage_contents.get(url, "Content not available")
        
        prompt += f"""
        {idx}. Title: {result['title']}
        URL: {url}
        Description: {result['description']}
        
        Content Summary:
        {content[:2000]}  # Limit content length to manage token count
        
        {'-' * 30}
        """
    
    prompt += f"""
    {'-' * 50}
    
    Please provide:
    1. A comprehensive summary of key findings and insights, explicitly citing sources inline (e.g., [Source 1]).
    2. A detailed breakdown of evidence with clear reference to the provided sources.
    3. Identification of conflicting perspectives or gaps in the research.
    4. Proactive recommendations for further investigation and potential new lines of inquiry.
    5. Clearly label any speculative observations.
    
    Format your response in clear markdown sections.
    """
    
    chat = model.start_chat(history=[])
    with st.spinner("Analyzing webpage contents..."):
        response = chat.send_message(prompt)
        return response.text

def refine_research_query(original_query: str) -> str:
    """
    Refine the research query into a comprehensive and detailed research query.
    """
    prompt = f"""
    You are an expert researcher. Refine the following research question to be a comprehensive research query for further investigation.
    
    Original Research Question: "{original_query}"
    
    The refined query should:
    - Clarify any vague terms or ambiguities.
    - Expand on the context and include potential angles of inquiry.
    - Break down the question into specific research components if applicable.
    - Provide clear research goals.
    
    Provide only the refined query in your response.
    """
    response = model.generate_content(prompt)
    refined_query = response.text.strip()
    return refined_query

def simplify_search_query(refined_query: str) -> str:
    """
    Convert the refined research query into a succinct web search query suitable for using as a search term.
    """
    prompt = f"""
    Convert the following refined research query into a concise web search query that focuses on the most relevant keywords:
    
    {refined_query}
    
    Provide only the concise search query in your response.
    """
    response = model.generate_content(prompt)
    search_query = response.text.strip()
    return search_query

def simplify_search_queries(refined_query: str) -> List[str]:
    """
    Convert the refined research query into 5 succinct web search queries suitable for using as search terms.
    """
    prompt = f"""
    Convert the following refined research query into 5 concise web search queries that focus on the most relevant keywords:
    
    {refined_query}
    
    Provide each search query on a separate line.
    """
    response = model.generate_content(prompt)
    search_queries = [line.strip() for line in response.text.strip().splitlines() if line.strip()]
    return search_queries

# Streamlit UI
st.title("🔍 Deep Research Assistant")
st.markdown("Powered by Brave Search and Google Gemini")

# Input section
with st.form("research_form"):
    query = st.text_input("Enter your research query:", placeholder="What would you like to research?")
    submitted = st.form_submit_button("Start Research")

if submitted and query:
    with st.spinner("Refining research query..."):
         refined_query = refine_research_query(query)
    st.markdown(f"**Refined Research Query:** {refined_query}")

    with st.spinner("Simplifying search queries..."):
         web_search_queries = simplify_search_queries(refined_query)
    st.markdown("**Web Search Queries:**")
    for q in web_search_queries:
         st.markdown(f"- {q}")

    with st.spinner("Searching and analyzing..."):
         all_results = []
         # Fixed number of search results per query: 5
         for q in web_search_queries:
              results = brave_search(q, 5)
              all_results.extend(results)
              time.sleep(1)  # Delay between API requests to avoid rate limiting
         st.session_state.search_results = all_results
        
         if st.session_state.search_results:
              st.session_state.analysis = analyze_with_gemini(refined_query, st.session_state.search_results)

# Display results
if st.session_state.search_results:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📚 Search Results")
        for idx, result in enumerate(st.session_state.search_results, 1):
            with st.expander(f"{idx}. {result['title']}"):
                st.write(result['description'])
                st.markdown(f"[Read more]({result['url']})")
    
    with col2:
        st.subheader("🤖 AI Analysis")
        st.markdown(st.session_state.analysis)

# Add export functionality
if st.session_state.analysis:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_filename = f"research_report_{timestamp}.md"
    
    export_content = f"""# Research Report: {query}
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Analysis
{st.session_state.analysis}

## Sources
"""
    for idx, result in enumerate(st.session_state.search_results, 1):
        export_content += f"\n{idx}. [{result['title']}]({result['url']})"

    st.download_button(
        label="📥 Export Report",
        data=export_content,
        file_name=export_filename,
        mime="text/markdown"
    ) 