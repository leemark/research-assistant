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

def generate_search_queries(query: str, learnings: List[str] = None, num_queries: int = 3) -> List[Dict]:
    """
    Generate multiple search queries to explore the topic more broadly
    """
    prompt = f"""
    Given the following research query, generate {num_queries} unique search queries to explore different aspects:
    Query: {query}
    
    {f'Previous learnings:\n{chr(10).join(learnings)}' if learnings else ''}
    
    Return each query with its research goal and follow-up directions.
    """
    
    response = model.generate_content(prompt)
    # Parse response to extract queries and goals
    # This is a simplified version - you might want to add more structure
    queries = []
    try:
        lines = response.text.split('\n')
        current_query = {}
        for line in lines:
            if line.startswith('Query:'):
                if current_query:
                    queries.append(current_query)
                current_query = {'query': line.replace('Query:', '').strip()}
            elif line.startswith('Goal:') and current_query:
                current_query['goal'] = line.replace('Goal:', '').strip()
        if current_query:
            queries.append(current_query)
    except Exception as e:
        st.error(f"Error parsing queries: {str(e)}")
        queries = [{'query': query, 'goal': 'Original query'}]
    
    return queries[:num_queries]

def process_search_results(query: str, search_results: List[Dict], webpage_contents: Dict[str, str], 
                         num_learnings: int = 3, num_followup: int = 3) -> Dict:
    """
    Process search results to extract learnings and follow-up questions
    """
    prompt = f"""
    Analyze the following search results for the query: "{query}"
    
    {'-' * 50}
    """
    
    for idx, result in enumerate(search_results, 1):
        content = webpage_contents.get(result['url'], "Content not available")
        prompt += f"""
        Source {idx}:
        Title: {result['title']}
        Content: {content[:2000]}
        
        {'-' * 30}
        """
    
    prompt += f"""
    Please provide:
    1. Top {num_learnings} key learnings (be specific, include metrics and entities)
    2. {num_followup} follow-up questions for deeper research
    """
    
    response = model.generate_content(prompt)
    
    # Parse response to extract learnings and questions
    try:
        sections = response.text.split('Follow-up questions:')
        learnings = [l.strip() for l in sections[0].split('\n') if l.strip()]
        questions = [q.strip() for q in sections[1].split('\n') if q.strip()]
        return {
            'learnings': learnings[:num_learnings],
            'followup_questions': questions[:num_followup]
        }
    except Exception as e:
        st.error(f"Error parsing results: {str(e)}")
        return {'learnings': [], 'followup_questions': []}

def deep_research(query: str, breadth: int = 3, depth: int = 2, 
                 learnings: List[str] = None, visited_urls: List[str] = None) -> Dict:
    """
    Perform recursive deep research with breadth and depth
    """
    if learnings is None:
        learnings = []
    if visited_urls is None:
        visited_urls = []
        
    search_queries = generate_search_queries(query, learnings, breadth)
    all_learnings = learnings.copy()
    all_urls = visited_urls.copy()
    
    for search_query in search_queries:
        try:
            results = brave_search(search_query['query'])
            new_urls = [r['url'] for r in results]
            webpage_contents = scrape_urls_parallel([url for url in new_urls if url not in all_urls])
            
            processed_results = process_search_results(
                search_query['query'], 
                results, 
                webpage_contents
            )
            
            all_learnings.extend(processed_results['learnings'])
            all_urls.extend(new_urls)
            
            if depth > 1:
                for followup in processed_results['followup_questions']:
                    deeper_results = deep_research(
                        followup,
                        breadth=max(2, breadth-1),
                        depth=depth-1,
                        learnings=all_learnings,
                        visited_urls=all_urls
                    )
                    all_learnings.extend(deeper_results['learnings'])
                    all_urls.extend(deeper_results['visited_urls'])
                    
        except Exception as e:
            st.error(f"Error in research: {str(e)}")
            continue
    
    return {
        'learnings': list(set(all_learnings)),
        'visited_urls': list(set(all_urls))
    }

def generate_feedback(query: str, num_questions: int = 3) -> List[str]:
    """
    Generate clarifying questions for the research query
    """
    prompt = f"""
    Given the following research query, generate {num_questions} clarifying questions 
    to better understand the research direction:
    
    Query: {query}
    
    The questions should help:
    - Narrow down the scope if too broad
    - Clarify ambiguous terms
    - Identify specific aspects of interest
    - Determine the desired depth of research
    """
    
    response = model.generate_content(prompt)
    
    try:
        questions = [q.strip() for q in response.text.split('\n') if q.strip()]
        return questions[:num_questions]
    except Exception as e:
        st.error(f"Error generating feedback: {str(e)}")
        return []

# Streamlit UI
st.title("🔍 Deep Research Assistant")
st.markdown("Powered by Brave Search and Google Gemini")

# Input section
with st.form("research_form"):
    query = st.text_input("Enter your research query:", placeholder="What would you like to research?")
    
    if query:
        feedback_questions = generate_feedback(query)
        if feedback_questions:
            st.subheader("📝 Before we begin, let's clarify:")
            for q in feedback_questions:
                answer = st.text_input(q)
                if answer:
                    query += f"\nContext - {q}: {answer}"
    
    col1, col2 = st.columns(2)
    with col1:
        breadth = st.slider("Research breadth:", min_value=2, max_value=5, value=3,
                          help="Number of parallel search queries")
    with col2:
        depth = st.slider("Research depth:", min_value=1, max_value=3, value=2,
                         help="Number of recursive follow-up rounds")
    
    submitted = st.form_submit_button("Start Deep Research")

if submitted and query:
    with st.spinner("Performing deep research..."):
        research_results = deep_research(query, breadth=breadth, depth=depth)
        
        st.session_state.search_results = research_results['visited_urls']
        st.session_state.learnings = research_results['learnings']
        
        # Generate final analysis
        st.session_state.analysis = analyze_with_gemini(
            query, 
            research_results['learnings'],
            research_results['visited_urls']
        )

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