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

def brave_search(query: str, num_results: int = 10) -> List[Dict]:
    """
    Perform a search using Brave Search API
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
    
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        results = response.json().get('web', {}).get('results', [])
        return results
    else:
        st.error(f"Search API Error: {response.status_code}")
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
    Analyze search results using Gemini 2.0 Flash Thinking with enhanced expert analysis
    """
    chat = model.start_chat(history=[])
    
    # Scrape webpage contents
    urls = [result['url'] for result in search_results]
    webpage_contents = scrape_urls_parallel(urls)
    
    current_time = datetime.now().isoformat()
    
    prompt = f"""
    You are an expert researcher and analyst. Today is {current_time}. 
    
    Core Analysis Guidelines:
    - Treat all information as if presented to a highly experienced analyst
    - Be extremely detailed and thorough in your analysis
    - Maintain rigorous accuracy and cite specific evidence from the sources
    - Focus on strong arguments and evidence over source authority
    - Consider contrarian viewpoints and challenge conventional wisdom
    - Flag any speculative conclusions or predictions clearly
    - Be proactive in identifying implications the user may not have considered
    - Organize information with clear structure and hierarchy
    
    Research Query: {query}
    
    Analyze the following search results and their full contents comprehensively:
    
    {'-' * 50}
    """
    
    for idx, result in enumerate(search_results, 1):
        url = result['url']
        content = webpage_contents.get(url, "Content not available")
        
        prompt += f"""
        Source {idx}:
        Title: {result['title']}
        URL: {url}
        Description: {result['description']}
        
        Full Content Analysis:
        {content[:2000]}
        
        {'-' * 30}
        """
    
    prompt += f"""
    {'-' * 50}
    
    Provide a comprehensive expert analysis covering:

    1. Core Findings
    - Synthesize key discoveries from all sources
    - Identify critical patterns and relationships
    - Highlight unexpected or notable findings
    
    2. Technical Analysis
    - Detailed examination of methodologies used
    - Evaluation of data quality and reliability
    - Technical limitations or constraints identified
    
    3. Competing Perspectives
    - Compare and contrast different viewpoints
    - Analyze conflicts in methodology or conclusions
    - Evaluate strength of supporting evidence
    
    4. Critical Gaps
    - Identify missing information or unexplored areas
    - Point out potential biases or limitations
    - Flag areas needing additional verification
    
    5. Strategic Implications
    - Long-term consequences and impacts
    - Potential future developments
    - Strategic recommendations
    
    6. Expert Recommendations
    - Specific actions based on findings
    - Priority areas for further investigation
    - Alternative approaches to consider
    
    7. Speculative Analysis
    - [SPECULATIVE] Clearly marked predictions or forecasts
    - Potential emerging trends
    - Alternative scenarios to consider
    
    Format your response using clear markdown structure with detailed subsections.
    Prioritize accuracy and depth over brevity.
    Challenge assumptions and consider non-obvious implications.
    """
    
    # Add progress indicator
    with st.spinner("Performing expert analysis of source materials..."):
        response = chat.send_message(prompt)
        return response.text

# Streamlit UI
st.title("🔍 Deep Research Assistant")
st.markdown("Powered by Brave Search and Google Gemini")

# Input section
with st.form("research_form"):
    query = st.text_input("Enter your research query:", placeholder="What would you like to research?")
    num_results = st.slider("Number of search results to analyze:", min_value=5, max_value=20, value=10)
    submitted = st.form_submit_button("Start Research")

if submitted and query:
    with st.spinner("Searching and analyzing..."):
        # Perform search
        st.session_state.search_results = brave_search(query, num_results)
        
        if st.session_state.search_results:
            # Analyze results
            st.session_state.analysis = analyze_with_gemini(query, st.session_state.search_results)

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