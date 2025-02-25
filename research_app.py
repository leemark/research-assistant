import streamlit as st
import requests
import google.generativeai as genai
from typing import List, Dict, Tuple, Optional
import os
from datetime import datetime
from bs4 import BeautifulSoup
import concurrent.futures
import time
from urllib.parse import urlparse
import json
import pathlib
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend which is non-interactive
import io
import base64
from streamlit_timeline import timeline

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure page settings
st.set_page_config(
    page_title="DEEP fREeSEARCH",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .research-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid #4169E1;
        color: #333333;
    }
    .section-card {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #5cb85c;
        color: #333333;
    }
    .plan-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
        border-left: 5px solid #ff7f0e;
        color: #333333;
    }
    .info-box {
        background-color: #e6f3ff;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        border: 1px solid #b8daff;
        color: #333333;
    }
    .sources-box {
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        max-height: 200px;
        overflow-y: auto;
        color: #333333;
    }
    .key-question {
        background-color: #f2f7ff;
        border-radius: 5px;
        padding: 5px 10px;
        margin: 5px 0;
        display: inline-block;
        font-size: 0.9em;
        color: #333333;
    }
    .phase-indicator {
        font-size: 0.8em;
        font-weight: bold;
        color: #6c757d;
        margin-bottom: 10px;
    }
    .progress-label {
        font-size: 0.9em;
        font-weight: bold;
        margin-bottom: 5px;
        color: #333333;
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .header-text {
        margin: 0;
        color: #333333;
    }
    .header-button {
        margin-left: 15px;
    }
    
    /* Enhanced Tab styling */
    .stTabs {
        margin-top: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background-color: #1e222e;
        padding: 10px 10px 0 10px;
        border-radius: 10px 10px 0 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 8px 8px 0 0;
        margin-right: 5px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
        background-color: rgba(255, 255, 255, 0.08);
        border: none;
        color: #a9b2c3;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #3d5afb;
        color: white;
        box-shadow: 0 2px 10px rgba(61, 90, 251, 0.5);
        transform: translateY(-5px);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        background-color: rgba(255, 255, 255, 0.15);
        color: white;
        transform: translateY(-3px);
    }
    .stTabs [data-baseweb="tab-panel"] {
        background-color: white;
        border-radius: 0 0 10px 10px;
        padding: 20px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        color: #333333;
    }
    .stTabs [data-baseweb="tab-panel"] p,
    .stTabs [data-baseweb="tab-panel"] h1,
    .stTabs [data-baseweb="tab-panel"] h2,
    .stTabs [data-baseweb="tab-panel"] h3,
    .stTabs [data-baseweb="tab-panel"] h4,
    .stTabs [data-baseweb="tab-panel"] li {
        color: #333333;
    }
    .section-header {
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 10px;
        color: #2c3e50;
    }
    
    /* Custom button styles */
    .custom-button {
        display: inline-block;
        padding: 8px 16px;
        font-weight: 600;
        text-align: center;
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.3s ease;
        margin: 5px 0;
        text-decoration: none;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .primary-button {
        background-color: #3d5afb;
        color: white !important;
        border: none;
    }
    .primary-button:hover {
        background-color: #2a41d8;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    .secondary-button {
        background-color: #f0f2f6;
        color: #5a6a85 !important;
        border: 1px solid #d8dde6;
    }
    .secondary-button:hover {
        background-color: #e0e5ed;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Custom tab icon styles */
    .tab-icon {
        margin-right: 6px;
        margin-top: -2px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for API keys
if 'api_keys_initialized' not in st.session_state:
    st.session_state.api_keys_initialized = False

# Function to load cached API keys
def load_cached_api_keys():
    try:
        cache_file = pathlib.Path('.streamlit/api_keys.json')
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
    except Exception:
        return None
    return None

# Function to save API keys to cache
def save_api_keys_to_cache(brave_key: str, google_key: str):
    try:
        cache_dir = pathlib.Path('.streamlit')
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / 'api_keys.json'
        with open(cache_file, 'w') as f:
            json.dump({
                'BRAVE_API_KEY': brave_key,
                'GOOGLE_API_KEY': google_key
            }, f)
    except Exception as e:
        st.error(f"Failed to cache API keys: {str(e)}")

# Load cached API keys if available
cached_keys = load_cached_api_keys()

# API Key Input Section
if not st.session_state.api_keys_initialized:
    st.title("🔍 DEEP fREeSEARCH")
    st.markdown("""
    Welcome to DEEP fREeSEARCH - your AI-powered research assistant! This tool helps you:
    - 🎯 Conduct comprehensive research on any topic
    - 🤖 Get AI-powered analysis of multiple sources
    - 📊 Generate detailed research reports automatically
    - 📥 Export findings in markdown format

    To get started, you'll need:
    1. A [Brave Search API key](https://api.search.brave.com/app/keys) (free tier available)
    2. A [Google API key](https://makersuite.google.com/app/apikey) for Gemini 2.0 Flash Thinking (free tier available)
    """)

    st.divider()
    
    st.title("🔑 API Key Setup")
    
    col1, col2 = st.columns(2)
    
    with col1:
        brave_key = st.text_input(
            "Enter your Brave API Key",
            value=cached_keys.get('BRAVE_API_KEY', '') if cached_keys else '',
            type="password"
        )
        st.markdown("[Get Brave API Key](https://api.search.brave.com/app/keys)")
    
    with col2:
        google_key = st.text_input(
            "Enter your Google API Key",
            value=cached_keys.get('GOOGLE_API_KEY', '') if cached_keys else '',
            type="password"
        )
        st.markdown("[Get Google API Key](https://makersuite.google.com/app/apikey)")
    
    save_keys = st.checkbox("Save API keys locally (they will be cached for future sessions)", value=True)
    
    if st.button("Submit API Keys"):
        if not brave_key or not google_key:
            st.error("Please enter both API keys")
        else:
            # Test the API keys before proceeding
            try:
                # Test Brave API
                headers = {
                    "X-Subscription-Token": brave_key,
                    "Accept": "application/json",
                }
                brave_response = requests.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    headers=headers,
                    params={"q": "test"}
                )
                brave_response.raise_for_status()
                
                # Test Google API
                genai.configure(api_key=google_key)
                model = genai.GenerativeModel('gemini-pro')
                model.generate_content("test")
                
                # If both tests pass, save the keys
                if save_keys:
                    save_api_keys_to_cache(brave_key, google_key)
                
                # Store in session state
                st.session_state.BRAVE_API_KEY = brave_key
                st.session_state.GOOGLE_API_KEY = google_key
                st.session_state.api_keys_initialized = True
                st.success("API keys validated successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error validating API keys: {str(e)}")
    
    st.stop()  # Don't show the rest of the app until API keys are set

# Configure API keys from session state
BRAVE_API_KEY = st.session_state.BRAVE_API_KEY
GOOGLE_API_KEY = st.session_state.GOOGLE_API_KEY

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize other session state variables
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'analysis' not in st.session_state:
    st.session_state.analysis = ""
if 'refined_query' not in st.session_state:
    st.session_state.refined_query = ""
if 'research_iterations' not in st.session_state:
    st.session_state.research_iterations = []
if 'knowledge_graph' not in st.session_state:
    st.session_state.knowledge_graph = {}
if 'current_iteration' not in st.session_state:
    st.session_state.current_iteration = 0
if 'research_plan' not in st.session_state:
    st.session_state.research_plan = None
if 'current_section' not in st.session_state:
    st.session_state.current_section = 0
if 'sections' not in st.session_state:
    st.session_state.sections = []
if 'section_data' not in st.session_state:
    st.session_state.section_data = {}
if 'section_sources' not in st.session_state:
    st.session_state.section_sources = {}
if 'research_phase' not in st.session_state:
    st.session_state.research_phase = "initial"
if 'show_completed_section' not in st.session_state:
    st.session_state.show_completed_section = None
if 'show_section_sources' not in st.session_state:
    st.session_state.show_section_sources = None
if 'edit_section_id' not in st.session_state:
    st.session_state.edit_section_id = None
if 'report_tab' not in st.session_state:
    st.session_state.report_tab = "plan"
if 'section_status' not in st.session_state:
    st.session_state.section_status = {}

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
    Analyze search results using Gemini 2.0 Flash Thinking
    """
    # Scrape webpage contents from provided sources
    urls = [result['url'] for result in search_results]
    webpage_contents = scrape_urls_parallel(urls)
    
    current_time = datetime.now().isoformat()
    prompt = f"""
    You are an expert researcher. Today is {current_time}. Please generate a comprehensive yet accessible research report based on the following instructions and sources.

    INSTRUCTIONS:
    - You may be asked to research subjects that are post your knowledge cutoff; assume the user is right when new information is presented.
    - The user is a highly experienced analyst, so be as detailed and accurate as possible.
    - Be highly organized, proactive, and anticipate further needs.
    - Suggest solutions or insights that might not have been considered.
    - Provide detailed explanations and analysis.
    - Value good arguments over mere authority; however, reference the provided sources explicitly using inline citations (e.g., [Source 1]).
    - Consider new technologies and contrarian ideas, and clearly flag any high levels of speculation.

    --------------------------------------------------
    Executive Summary:
    - Provide one combined executive summary for the entire research report using 3-5 bullet points. 
    - Ensure the summary captures the most critical findings and recommendations across all sources.
    --------------------------------------------------

    Following the Executive Summary, provide a detailed analysis that includes:
    1. A robust breakdown of evidence with source citations.
    2. Identification of conflicting perspectives or gaps in the research.
    3. Proactive recommendations for further investigation.
    4. Clearly flag any speculative observations.

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
        {content[:4000]}  # Limit content length to manage token count
        
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
    with st.spinner("Analyzing search results..."):
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

def simplify_search_query(refined_query: str) -> List[str]:
    """
    Convert the refined research query into 5 different succinct web search queries suitable for using as search terms.
    Returns a list of 5 different search queries.
    """
    prompt = f"""
    Convert the following refined research query into 5 different concise web search queries that focus on different aspects or angles of the research topic. Make each query unique and focused on a different perspective or subtopic:

    {refined_query}

    Provide exactly 5 different search queries, one per line, with no additional text or formatting.
    """
    response = model.generate_content(prompt)
    search_queries = [q.strip() for q in response.text.strip().split('\n') if q.strip()]
    return search_queries[:5]  # Ensure we return exactly 5 queries

def to_headline_case(text: str) -> str:
    """Convert text to headline case, properly handling articles, conjunctions, and prepositions."""
    # Words that should not be capitalized (unless they're the first word)
    minor_words = {'a', 'an', 'and', 'as', 'at', 'but', 'by', 'for', 'in', 'of', 'on', 'or', 'the', 'to', 'with'}
    
    words = text.split()
    if not words:
        return text
        
    # Capitalize the first word regardless of what it is
    result = [words[0].capitalize()]
    
    # Process the rest of the words
    for word in words[1:]:
        if word.lower() in minor_words:
            result.append(word.lower())
        else:
            result.append(word.capitalize())
            
    return ' '.join(result)

def generate_research_title(query: str) -> str:
    """
    Generate a concise, professional research title from the query.
    Uses Gemini to create an appropriate academic-style title.
    """
    prompt = f"""
    Create a concise, professional research title from this query. The title should:
    - Be clear and academic in style
    - Not exceed 12 words
    - Capture the main research focus
    - Not use unnecessary words like "Research on" or "Investigation of"
    - Use proper title case
    
    Query: {query}
    
    Provide only the title with no additional text or punctuation.
    """
    
    try:
        response = model.generate_content(prompt)
        title = response.text.strip()
        return title
    except Exception as e:
        # Fallback to a simpler title if generation fails
        return f"Analysis of {query[:100]}"

def write_final_report(refined_query: str, analyses: List[Tuple[str, str]], search_results: List[Dict], initial_report: str) -> str:
    """
    Generate a final combined research report using the initial analysis as context.
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Generate a better title
    title = generate_research_title(refined_query)
    
    # Build the final report prompt
    prompt = f"""
    You are an expert research writer. Draft a comprehensive research report that focuses on the subject matter rather than the research process.
    IMPORTANT: Follow these guidelines:
    1. Focus on the research topic and findings, NOT on how the research was conducted
    2. Do not mention the tool, iterations, or AI unless they are specifically part of the research topic
    3. Use meaningful section titles that reflect the actual content
    4. Maintain academic tone and style
    5. Include specific examples and evidence
    6. The title and timestamp must be formatted exactly as shown below:

    # {title}
    *Generated on: {current_time}*

    ## Table of Contents
    1. [Abstract](#abstract)
    2. [Research Questions](#research-questions)
    3. [Executive Summary](#executive-summary)
    4. [Key Findings and Analysis](#key-findings-and-analysis)
    5. [Implications and Applications](#implications-and-applications)
    6. [Challenges and Limitations](#challenges-and-limitations)
    7. [Future Directions](#future-directions)
    8. [Sources Analyzed](#sources-analyzed)

    ## Abstract
    Provide a concise summary of the key findings and implications, focusing on the research topic itself.

    ## Research Questions
    {refined_query}

    ## Executive Summary
    Produce a focused executive summary using 5-8 bullet points that capture the most important findings and implications.
    Do not discuss the research process unless it's specifically part of the research topic.

    ## Key Findings and Analysis
    Present the main findings and analysis, organized by major themes or topics.
    Focus on the subject matter and evidence rather than how the information was gathered.

    ## Implications and Applications
    Discuss the practical and theoretical implications of the findings.
    Include specific examples and potential applications.

    ## Challenges and Limitations
    Address key challenges, limitations, and areas of uncertainty in the subject matter.
    Focus on limitations in the field/topic being researched, not limitations of the research process.

    ## Future Directions
    Outline promising areas for future research and development in this field.
    Focus on advancing the subject matter rather than improving research methodology.

    ## Sources Analyzed
    """
    
    # Append the sources list
    for idx, result in enumerate(search_results, 1):
         prompt += f"\n{idx}. [{result['title']}]({result['url']})"
    
    prompt += "\n\nPlease produce the final report in clear markdown format with the above structure. Do not put quotes around the Research Questions content."

    # Start a new Gemini chat and use the prompt
    chat = model.start_chat(history=[])
    with st.spinner("Generating final report..."):
         final_response = chat.send_message(prompt)
         return final_response.text

def identify_knowledge_gaps(analysis: str) -> List[str]:
    """
    Analyze the current research findings to identify knowledge gaps and areas needing deeper investigation.
    Returns a list of specific questions or topics that need further research.
    """
    prompt = f"""
    As an expert researcher, analyze the following research findings and identify specific knowledge gaps 
    or areas that require deeper investigation. Focus on:
    1. Unanswered questions in the current analysis
    2. Conflicting information that needs resolution
    3. Topics mentioned but not fully explored
    4. Potential connections or implications not yet investigated
    5. Missing context or background information

    Current Analysis:
    {analysis}

    Return ONLY a list of 3-5 specific questions or topics that need further investigation.
    Each item should be clear and focused enough to serve as a new search query.
    Format: One item per line, no numbering or bullets.
    """
    
    response = model.generate_content(prompt)
    gaps = [gap.strip() for gap in response.text.strip().split('\n') if gap.strip()]
    return gaps[:5]  # Ensure we return at most 5 gaps

def synthesize_iterations(iterations: List[Dict]) -> str:
    """
    Synthesize findings across multiple research iterations into a cohesive analysis.
    Each iteration contains the original query, search results, and analysis.
    """
    prompt = f"""
    As an expert researcher, synthesize the findings from multiple research iterations into a comprehensive analysis.
    Focus on:
    1. How later iterations built upon or refined earlier findings
    2. Resolution of knowledge gaps identified in earlier iterations
    3. Evolution of understanding across iterations
    4. Emerging patterns or themes
    5. Remaining open questions or areas for future research

    Research Iterations:
    {'-' * 50}
    """
    
    for idx, iteration in enumerate(iterations, 1):
        prompt += f"""
        Iteration {idx}:
        Query: {iteration['query']}
        Analysis Summary: {iteration['analysis'][:2000]}  # Limit length for token management
        Knowledge Gaps Addressed: {', '.join(iteration.get('gaps_addressed', []))}
        
        {'-' * 30}
        """
    
    prompt += """
    Please provide:
    1. A synthesis of how understanding evolved across iterations
    2. Key insights that emerged from the iterative process
    3. How knowledge gaps were addressed
    4. Remaining open questions
    5. Recommendations for future research
    
    Format your response in clear markdown sections.
    """
    
    chat = model.start_chat(history=[])
    response = chat.send_message(prompt)
    return response.text

def update_knowledge_graph(new_content: str, current_graph: Dict) -> Dict:
    """
    Update the knowledge graph based on new content.
    Returns the updated graph.
    """
    prompt = f"""
    You are an expert knowledge mapper. Analyze the following research content and extract a knowledge graph:
    
    {new_content[:10000]}  # Limit content length to manage token count
    
    The knowledge graph should represent key concepts and their relationships.
    For each concept, track evidence supporting or conflicting with it.
    
    Return a JSON object with this exact structure:
    {{
      "concepts": {{
        "concept_name": {{
          "confidence": float,  # 0.0 to 1.0
          "supporting_evidence": [string],
          "conflicting_evidence": [string]
        }},
        ...more concepts...
      }},
      "relationships": [
        {{
          "source": "concept_name_1",
          "target": "concept_name_2",
          "type": "relationship_type",
          "confidence": float  # 0.0 to 1.0
        }},
        ...more relationships...
      ]
    }}
    
    Extract 5-15 key concepts and their relationships. Focus on the most important concepts and strongest relationships.
    Include ONLY JSON in your response with no other text.
    """
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Try to find JSON content
        try:
            start_idx = response_text.index('{')
            end_idx = response_text.rindex('}') + 1
            json_str = response_text[start_idx:end_idx]
        except ValueError:
            # If we can't extract clean JSON, return the current graph
            st.warning("Failed to parse knowledge graph structure.")
            return current_graph
            
        try:
            new_graph = json.loads(json_str)
            
            # Merge with existing graph
            if not current_graph.get('concepts'):
                current_graph['concepts'] = {}
            if not current_graph.get('relationships'):
                current_graph['relationships'] = []
                
            # Update concepts
            for concept, data in new_graph.get('concepts', {}).items():
                if concept in current_graph['concepts']:
                    # Update existing concept
                    current_concept = current_graph['concepts'][concept]
                    current_concept['confidence'] = (current_concept.get('confidence', 0) + data.get('confidence', 0)) / 2
                    
                    # Merge evidence
                    if 'supporting_evidence' not in current_concept:
                        current_concept['supporting_evidence'] = []
                    if 'conflicting_evidence' not in current_concept:
                        current_concept['conflicting_evidence'] = []
                        
                    for evidence in data.get('supporting_evidence', []):
                        if evidence not in current_concept['supporting_evidence']:
                            current_concept['supporting_evidence'].append(evidence)
                            
                    for evidence in data.get('conflicting_evidence', []):
                        if evidence not in current_concept['conflicting_evidence']:
                            current_concept['conflicting_evidence'].append(evidence)
                else:
                    # Add new concept
                    current_graph['concepts'][concept] = data
            
            # Update relationships
            for rel in new_graph.get('relationships', []):
                # Check if relationship already exists
                exists = False
                for existing_rel in current_graph['relationships']:
                    if (existing_rel.get('source') == rel.get('source') and 
                        existing_rel.get('target') == rel.get('target') and
                        existing_rel.get('type') == rel.get('type')):
                        # Update confidence
                        existing_rel['confidence'] = (existing_rel.get('confidence', 0) + rel.get('confidence', 0)) / 2
                        exists = True
                        break
                
                # Add new relationship if it doesn't exist
                if not exists:
                    current_graph['relationships'].append(rel)
            
            return current_graph
            
        except json.JSONDecodeError:
            st.warning("Failed to parse knowledge graph JSON.")
            return current_graph
            
    except Exception as e:
        st.warning(f"Error updating knowledge graph: {str(e)}")
        return current_graph

def determine_next_iteration(knowledge_gaps: List[str], knowledge_graph: Dict) -> Tuple[bool, str]:
    """
    Determine if another research iteration is needed and what it should focus on.
    Returns a tuple of (should_continue: bool, next_query: str)
    """
    prompt = f"""
    Analyze the current research state and determine if another iteration would be valuable.
    Consider:
    1. Identified knowledge gaps: {knowledge_gaps}
    2. Current knowledge confidence levels from graph
    3. Potential for new insights
    4. Diminishing returns

    Knowledge Graph Summary:
    {json.dumps(knowledge_graph, indent=2)}

    Return a JSON object:
    {{
        "should_continue": boolean,
        "next_query": string,  # Most promising gap to investigate next, or empty if should_continue is false
        "reasoning": string    # Brief explanation of the decision
    }}
    """
    
    response = model.generate_content(prompt)
    try:
        decision = json.loads(response.text)
        return decision["should_continue"], decision["next_query"]
    except (json.JSONDecodeError, KeyError):
        return False, ""

def generate_research_plan(query: str) -> Dict:
    """
    Generate a comprehensive research plan with sections for the given query.
    Returns a dictionary with the plan structure.
    """
    prompt = f"""
    You are an expert research planner. Create a comprehensive research plan for investigating the following topic:
    
    Topic: "{query}"
    
    Your task is to:
    1. Break down this topic into 4-7 clearly defined sections that together will provide thorough coverage
    2. For each section, provide a brief description of what should be researched
    3. Identify the logical ordering of sections that builds understanding progressively
    4. Consider both breadth (covering different aspects) and depth (exploring key areas thoroughly)
    5. Flag any sections that may require specialized knowledge or more extensive research
    
    Return your plan as a JSON object with exactly this structure:
    {{
      "title": "Main research report title",
      "sections": [
        {{
          "id": 1,
          "title": "Section title",
          "description": "Brief description of research goals for this section",
          "key_questions": ["Question 1", "Question 2", "Question 3"]
        }},
        ...additional sections...
      ],
      "introduction": "Brief overview of the research approach",
      "conclusion": "What the final synthesis should address"
    }}

    Your response should ONLY include the JSON object with no additional text.
    """
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Try to find JSON content if there's any surrounding text
        try:
            start_idx = response_text.index('{')
            end_idx = response_text.rindex('}') + 1
            json_str = response_text[start_idx:end_idx]
        except ValueError:
            st.warning("Failed to extract JSON from response.")
            return {
                "title": f"Research on {query}",
                "sections": [{"id": 1, "title": "General Overview", "description": "General information about the topic", "key_questions": [f"What is {query}?"]}],
                "introduction": "Research introduction",
                "conclusion": "Research conclusion"
            }
            
        try:
            research_plan = json.loads(json_str)
            return research_plan
        except json.JSONDecodeError as e:
            st.warning(f"Failed to parse research plan JSON: {str(e)}")
            return {
                "title": f"Research on {query}",
                "sections": [{"id": 1, "title": "General Overview", "description": "General information about the topic", "key_questions": [f"What is {query}?"]}],
                "introduction": "Research introduction",
                "conclusion": "Research conclusion"
            }
            
    except Exception as e:
        st.warning(f"Error generating research plan: {str(e)}")
        return {
            "title": f"Research on {query}",
            "sections": [{"id": 1, "title": "General Overview", "description": "General information about the topic", "key_questions": [f"What is {query}?"]}],
            "introduction": "Research introduction",
            "conclusion": "Research conclusion"
        }

def generate_section_queries(section: Dict, main_query: str) -> List[str]:
    """
    Generate search queries specifically for a section based on its content and key questions.
    Returns a list of search queries.
    """
    prompt = f"""
    You are an expert search query generator. Create effective search queries for researching the following section of a larger research project:
    
    Main Research Topic: "{main_query}"
    
    Section Title: "{section['title']}"
    Section Description: "{section['description']}"
    Key Questions:
    {json.dumps(section.get('key_questions', []))}
    
    Generate 3-5 search queries that will:
    1. Target specific information needed for this section
    2. Use varied approaches to find different aspects of the topic
    3. Be specific enough to return relevant results
    4. Include any specialized terminology that would help find authoritative sources
    5. Be properly formatted for web search (clear, concise, well-structured)
    
    Return ONLY a list of search queries, one per line, with no additional text, numbering, or formatting.
    """
    
    response = model.generate_content(prompt)
    search_queries = [q.strip() for q in response.text.strip().split('\n') if q.strip()]
    return search_queries[:5]  # Ensure we return at most 5 queries

def research_section(section: Dict, main_query: str) -> Dict:
    """
    Conduct research for a specific section of the report.
    Returns a dictionary with the section content and sources.
    """
    section_id = section['id']
    section_title = section['title']
    
    # If we've already researched this section, return the cached data
    if section_id in st.session_state.section_data:
        return {
            "content": st.session_state.section_data[section_id],
            "sources": st.session_state.section_sources.get(section_id, [])
        }
    
    # Generate search queries for this section
    search_queries = generate_section_queries(section, main_query)
    
    # Initialize collection of results and sources
    all_results = []
    all_sources = []
    
    # Conduct searches for each query
    for query in search_queries:
        search_results = brave_search(query)
        
        if search_results:
            all_results.extend(search_results)
            
            # Track sources for citation
            for result in search_results:
                all_sources.append({
                    "title": result['title'],
                    "url": result['url'],
                    "domain": urlparse(result['url']).netloc,
                    "query": query
                })
    
    # Remove duplicate sources
    unique_sources = []
    seen_urls = set()
    for source in all_sources:
        if source['url'] not in seen_urls:
            seen_urls.add(source['url'])
            unique_sources.append(source)
    
    # Analyze the search results for this section
    section_content = analyze_section(section, main_query, all_results, unique_sources)
    
    # Cache the results
    st.session_state.section_data[section_id] = section_content
    st.session_state.section_sources[section_id] = unique_sources
    
    return {
        "content": section_content,
        "sources": unique_sources
    }

def analyze_section(section: Dict, main_query: str, search_results: List[Dict], sources: List[Dict]) -> str:
    """
    Analyze search results specifically for a section of the report.
    Returns the section content with proper citations.
    """
    # Scrape webpage contents from provided sources
    urls = [result['url'] for result in search_results]
    webpage_contents = scrape_urls_parallel(urls)
    
    # Prepare the prompt for section analysis
    current_time = datetime.now().isoformat()
    prompt = f"""
    You are an expert researcher focusing on a specific section of a larger research report. 
    Today is {current_time}. 
    
    RESEARCH CONTEXT:
    Main Research Topic: "{main_query}"
    Current Section: "{section['title']}"
    Section Purpose: "{section['description']}"
    Key Questions for this Section:
    {json.dumps(section.get('key_questions', []))}
    
    INSTRUCTIONS:
    - Focus ONLY on the specific section assigned to you, not the entire research topic
    - Use inline citations to reference sources (e.g., [Source 1])
    - Prioritize depth over breadth for this specific section
    - Include key data, statistics, or findings relevant to this section
    - Address contradictions or debates in the literature if present
    - Maintain an objective, scholarly tone
    - Structure your response with clear subheadings appropriate for this section
    - Do NOT include general introductions or conclusions about the overall topic
    - IMPORTANT: Do NOT include the section title "{section['title']}" as a heading in your response - the title will be added automatically
    
    Below are the sources and their content summaries for this section:
    {'-' * 50}
    """
    
    for idx, result in enumerate(search_results, 1):
        url = result['url']
        content = webpage_contents.get(url, "Content not available")
        
        prompt += f"""
        {idx}. Title: {result['title']}
        URL: {url}
        Description: {result.get('description', 'No description available')}
        
        Content Summary:
        {content[:4000]}  # Limit content length to manage token count
        
        {'-' * 30}
        """
    
    prompt += f"""
    {'-' * 50}
    
    Write a comprehensive analysis ONLY for the "{section['title']}" section of the research report.
    Focus exclusively on this section's scope as defined above.
    Use proper markdown formatting with subheadings and lists as appropriate.
    Include inline citations [Source X] when referencing specific information.
    Do not include an introduction or conclusion to the overall report - just focus on this section.
    IMPORTANT: Do NOT include the section title "{section['title']}" as a heading in your response. The heading will be added automatically.
    """
    
    # Generate the section content
    chat = model.start_chat(history=[])
    with st.spinner(f"Analyzing sources for section: {section['title']}"):
        response = chat.send_message(prompt)
        return response.text

def synthesize_sections(sections: List[Dict], section_data: Dict, main_query: str, introduction: str, conclusion: str) -> str:
    """
    Synthesize all the section content into a cohesive report.
    """
    # Prepare the sections in order
    ordered_content = []
    for section in sections:
        section_id = section['id']
        if section_id in section_data:
            ordered_content.append({
                "title": section['title'],
                "content": section_data[section_id]
            })
    
    # Generate introduction if needed
    intro_prompt = f"""
    You are an expert researcher writing the introduction to a comprehensive research report.
    
    Research Topic: "{main_query}"
    
    The report contains the following sections:
    {json.dumps([s['title'] for s in sections])}
    
    Write a compelling introduction for this research report that:
    1. Clearly states the purpose and scope of the research
    2. Provides context for why this topic is important or relevant
    3. Briefly outlines the structure of the report and what readers will learn
    4. Sets the appropriate academic or professional tone
    5. Is approximately 250-350 words in length
    
    Your introduction should be engaging yet scholarly, and should not include citations.
    Format your response in clear markdown.
    """
    
    # Generate conclusion if needed
    conclusion_prompt = f"""
    You are an expert researcher writing the conclusion to a comprehensive research report.
    
    Research Topic: "{main_query}"
    
    The report contains the following sections:
    {json.dumps([{"title": s['title'], "key_points": section_data.get(s['id'], "")[:500]} for s in sections])}
    
    Write a thoughtful conclusion for this research report that:
    1. Synthesizes the key findings across all sections
    2. Identifies important patterns, trends, or connections
    3. Discusses limitations of the current research where appropriate
    4. Suggests areas for future research or exploration
    5. Ends with a strong closing statement about the significance of this topic
    6. Is approximately 250-350 words in length
    
    Your conclusion should provide valuable closure and perspective without introducing new information.
    Format your response in clear markdown.
    """
    
    # Generate final report with all sections
    chat = model.start_chat(history=[])
    
    # Generate introduction if not provided
    with st.spinner("Generating introduction..."):
        if not introduction or introduction == "Research introduction":
            intro_response = chat.send_message(intro_prompt)
            introduction = intro_response.text
    
    # Generate conclusion if not provided
    with st.spinner("Generating conclusion..."):
        if not conclusion or conclusion == "Research conclusion":
            conclusion_response = chat.send_message(conclusion_prompt)
            conclusion = conclusion_response.text
    
    # Assemble the final report
    final_report = f"# {main_query}\n\n"
    final_report += f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    
    # Add table of contents
    final_report += "## Table of Contents\n"
    final_report += "1. [Introduction](#introduction)\n"
    for i, section in enumerate(sections, 2):
        # Create anchor from title
        anchor = section['title'].lower().replace(' ', '-').replace(',', '').replace('.', '')
        final_report += f"{i}. [{section['title']}](#{anchor})\n"
    final_report += f"{len(sections) + 2}. [Conclusion](#conclusion)\n"
    final_report += f"{len(sections) + 3}. [Sources](#sources)\n\n"
    
    # Add introduction
    final_report += "## Introduction\n\n"
    final_report += introduction + "\n\n"
    
    # Add each section
    for section in sections:
        section_id = section['id']
        if section_id in section_data:
            final_report += f"## {section['title']}\n\n"
            final_report += section_data[section_id] + "\n\n"
    
    # Add conclusion
    final_report += "## Conclusion\n\n"
    final_report += conclusion + "\n\n"
    
    # Add sources
    final_report += "## Sources\n\n"
    all_sources = []
    for section_id, sources in st.session_state.section_sources.items():
        for source in sources:
            if source not in all_sources:
                all_sources.append(source)
    
    for i, source in enumerate(all_sources, 1):
        final_report += f"{i}. [{source['title']}]({source['url']})\n"
    
    return final_report

def render_knowledge_graph():
    """
    Visualize the knowledge graph with matplotlib and return the image as base64.
    """
    if not st.session_state.knowledge_graph or 'concepts' not in st.session_state.knowledge_graph:
        return None
        
    # Create a graph
    G = nx.Graph()
    
    # Add nodes (concepts)
    for concept, data in st.session_state.knowledge_graph.get('concepts', {}).items():
        # Use confidence as node size
        size = data.get('confidence', 0.5) * 1000
        G.add_node(concept, size=size)
    
    # Add edges (relationships)
    for rel in st.session_state.knowledge_graph.get('relationships', []):
        source = rel.get('source')
        target = rel.get('target')
        weight = rel.get('confidence', 0.5)
        if source in G.nodes and target in G.nodes:
            G.add_edge(source, target, weight=weight)
    
    if not G.nodes:
        return None
    
    # Create the visualization
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    
    # Get node sizes
    node_sizes = [G.nodes[node].get('size', 300) for node in G.nodes]
    
    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="#4169E1", alpha=0.8)
    
    # Edge weights as line thickness and alpha
    edge_weights = [G[u][v].get('weight', 0.5) * 3 for u, v in G.edges]
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7, edge_color="#5c5c5c")
    
    # Labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    
    plt.title("Knowledge Graph: Concept Relationships")
    plt.axis("off")
    
    # Convert the plot to an image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    plt.close()
    buf.seek(0)
    
    # Convert to base64 for display
    data = base64.b64encode(buf.read()).decode("utf-8")
    return data

def create_research_timeline():
    """
    Create a timeline of research events for visualization.
    """
    events = []
    
    # Add timeline entry for research plan
    if st.session_state.research_plan:
        events.append({
            "start_date": {
                "year": datetime.now().year,
                "month": datetime.now().month,
                "day": datetime.now().day,
                "hour": datetime.now().hour,
                "minute": datetime.now().minute,
            },
            "text": {
                "headline": "Research Plan Created",
                "text": f"<p>Created research plan for topic: {st.session_state.research_plan.get('title', '')}</p>"
            }
        })
    
    # Add timeline entries for each completed section
    for section_id, content in st.session_state.section_data.items():
        # Find the section with this ID
        section = next((s for s in st.session_state.sections if s['id'] == section_id), None)
        if section:
            # Create a timestamp slightly later than the previous event
            events.append({
                "start_date": {
                    "year": datetime.now().year,
                    "month": datetime.now().month,
                    "day": datetime.now().day,
                    "hour": datetime.now().hour,
                    "minute": datetime.now().minute + len(events),
                    "second": datetime.now().second,
                },
                "text": {
                    "headline": f"Researched: {section['title']}",
                    "text": f"<p>Completed research for section: {section['title']}</p><p>{section['description']}</p>"
                }
            })
    
    # Add final timeline entry if report is complete
    if st.session_state.research_phase == "complete" and hasattr(st.session_state, 'final_report'):
        events.append({
            "start_date": {
                "year": datetime.now().year,
                "month": datetime.now().month,
                "day": datetime.now().day,
                "hour": datetime.now().hour,
                "minute": datetime.now().minute + len(events),
            },
            "text": {
                "headline": "Final Report Generated",
                "text": "<p>Completed synthesis of all sections into final research report</p>"
            }
        })
    
    # Format the timeline data
    timeline_data = {
        "title": {
            "text": {
                "headline": "Research Process Timeline",
                "text": "<p>Key events in the research process</p>"
            }
        },
        "events": events
    }
    
    return timeline_data

def toggle_section_display(section_id):
    """
    Toggle the display of a completed section
    """
    if st.session_state.show_completed_section == section_id:
        st.session_state.show_completed_section = None
    else:
        st.session_state.show_completed_section = section_id

def toggle_sources_display(section_id):
    """
    Toggle the display of sources for a section
    """
    if st.session_state.show_section_sources == section_id:
        st.session_state.show_section_sources = None
    else:
        st.session_state.show_section_sources = section_id

def set_section_for_editing(section_id):
    """
    Set a section for editing
    """
    st.session_state.edit_section_id = section_id

def update_section_content(section_id, new_content):
    """
    Update the content of a section
    """
    if section_id in st.session_state.section_data:
        st.session_state.section_data[section_id] = new_content

# Streamlit UI
st.title("DEEP fREeSEARCH")
st.markdown("*The AI-powered research assistant*")

# Input section
with st.form("research_form"):
    query = st.text_input("Enter your research query:", placeholder="What would you like to research?")
    submitted = st.form_submit_button("Generate Research Plan")

# Reset research state if starting new query
if submitted and query and st.session_state.research_phase == "complete":
    st.session_state.research_plan = None
    st.session_state.current_section = 0
    st.session_state.sections = []
    st.session_state.section_data = {}
    st.session_state.section_sources = {}
    st.session_state.research_phase = "initial"
    st.session_state.current_iteration = 0
    st.session_state.research_iterations = []
    st.session_state.knowledge_graph = {}
    st.session_state.report_tab = "plan"

# Create tabs for different phases of research
if submitted and query or st.session_state.research_phase != "initial":
    # Create tabs with icons
    tab_icons = {
        "Research Plan": "📋", 
        "Research Progress": "📊", 
        "Knowledge Graph": "🔄", 
        "Final Report": "📄"
    }
    
    # Use the current tab from session state (default to "plan" if not set)
    active_tab_key = st.session_state.get("report_tab", "plan")
    tab_index_map = {"plan": 0, "progress": 1, "graph": 2, "report": 3}
    active_tab_index = tab_index_map.get(active_tab_key, 0)
    
    # Create tabs with the active tab pre-selected
    tabs = st.tabs([f"{icon} {name}" for name, icon in tab_icons.items()], index=active_tab_index)
    plan_tab, progress_tab, graph_tab, report_tab = tabs
    
    # PHASE 1: PLANNING
    with plan_tab:
        if st.session_state.research_phase == "initial":
            with st.spinner("Generating research plan..."):
                research_plan = generate_research_plan(query)
                st.session_state.research_plan = research_plan
                st.session_state.sections = research_plan["sections"]
                st.session_state.research_phase = "planning"
                st.rerun()  # Rerun to update the UI
        
        # Display the research plan for review if in planning phase
        if st.session_state.research_phase == "planning":
            st.markdown("## Research Plan")
            st.markdown(f"""<div class='plan-card'>
                <h3>{st.session_state.research_plan['title']}</h3>
                <p>{st.session_state.research_plan.get('introduction', '')}</p>
            </div>""", unsafe_allow_html=True)
            
            st.markdown("### Research Sections")
            for section in st.session_state.research_plan["sections"]:
                st.markdown(f"""<div class='section-card'>
                    <div class='section-header'>{section['title']}</div>
                    <p>{section['description']}</p>
                    <div style='margin-top: 10px;'>Key questions:</div>
                </div>""", unsafe_allow_html=True)
                
                for q in section.get('key_questions', []):
                    st.markdown(f"<div class='key-question'>{q}</div>", unsafe_allow_html=True)
            
            # Add custom styling for the buttons
            st.markdown("""
            <style>
            div[data-testid="stHorizontalBlock"] {
                background-color: #f8f9fa;
                border-radius: 10px;
                padding: 15px;
                margin-top: 20px;
                border-left: 5px solid #ff7f0e;
            }
            div.stButton > button {
                width: 100%;
                border-radius: 6px;
                font-weight: 600;
                padding: 0.5rem 1rem;
                transition: all 0.3s ease;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            }
            /* Style for the first column's button (Proceed) */
            div[data-testid="column"]:first-child div.stButton > button {
                background-color: #3d5afb;
                color: white;
                border: none;
            }
            div[data-testid="column"]:first-child div.stButton > button:hover {
                background-color: #2a41d8;
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
            }
            /* Style for the second column's button (Regenerate) */
            div[data-testid="column"]:nth-child(2) div.stButton > button {
                background-color: #f0f2f6;
                color: #5a6a85;
                border: 1px solid #d8dde6;
            }
            div[data-testid="column"]:nth-child(2) div.stButton > button:hover {
                background-color: #e0e5ed;
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Buttons for plan feedback
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Proceed with this plan", use_container_width=True):
                    # Save the current plan and prepare for research
                    st.session_state.research_phase = "researching"
                    st.session_state.report_tab = "progress"
                    
                    # Display a more prominent success message
                    st.markdown("""
                    <style>
                    .research-start-success {
                        background-color: #d4edda;
                        color: #155724;
                        padding: 15px;
                        border-radius: 8px;
                        border-left: 5px solid #28a745;
                        margin: 20px 0;
                        font-weight: 500;
                        animation: fadeIn 0.5s ease-in-out;
                    }
                    @keyframes fadeIn {
                        0% { opacity: 0; transform: translateY(10px); }
                        100% { opacity: 1; transform: translateY(0); }
                    }
                    </style>
                    <div class="research-start-success">
                        <span style="font-size: 1.1em;">✅ Starting research with approved plan!</span>
                        <div style="font-size: 0.9em; margin-top: 5px;">Switching to Research Progress tab...</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Short delay to show the success message and allow animation to play
                    time.sleep(1.5)
                    st.rerun()
            with col2:
                if st.button("Regenerate plan", use_container_width=True):
                    st.session_state.research_plan = None
                    st.session_state.research_phase = "initial"
                    st.info("Regenerating research plan...")
                    time.sleep(1)  # Short delay to show the info message
                    st.rerun()
        
        # Display the research plan in view-only mode for other research phases
        elif st.session_state.research_phase in ["researching", "synthesizing", "complete"] and st.session_state.research_plan:
            # Show a status indicator
            phase_status = {
                "researching": "🔍 Research in progress...",
                "synthesizing": "🔄 Synthesizing results...",
                "complete": "✅ Research complete!"
            }
            st.markdown(f"""
            <div style="background-color: #f2f7ff; border-radius: 5px; padding: 10px; margin-bottom: 20px; border-left: 4px solid #3d5afb;">
                <strong>{phase_status.get(st.session_state.research_phase, "")}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Display the research plan in view-only mode
            st.markdown("## Research Plan")
            st.markdown(f"""<div class='plan-card'>
                <h3>{st.session_state.research_plan['title']}</h3>
                <p>{st.session_state.research_plan.get('introduction', '')}</p>
            </div>""", unsafe_allow_html=True)
            
            st.markdown("### Research Sections")
            for section in st.session_state.research_plan["sections"]:
                # Add status indicator for each section
                section_id = section['id']
                if section_id in st.session_state.section_data:
                    status_icon = "✅"
                    status_color = "#28a745"
                elif section_id == st.session_state.current_section:
                    status_icon = "🔍"
                    status_color = "#17a2b8"
                else:
                    status_icon = "⏳"
                    status_color = "#6c757d"
                
                st.markdown(f"""<div class='section-card'>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div class='section-header'>{section['title']}</div>
                        <div style="color: {status_color}; font-weight: bold;">{status_icon}</div>
                    </div>
                    <p>{section['description']}</p>
                    <div style='margin-top: 10px;'>Key questions:</div>
                </div>""", unsafe_allow_html=True)
                
                for q in section.get('key_questions', []):
                    st.markdown(f"<div class='key-question'>{q}</div>", unsafe_allow_html=True)
    
    # PHASE 2: SECTION-BY-SECTION RESEARCH
    with progress_tab:
        if st.session_state.research_phase in ["researching", "synthesizing", "complete"]:
            # Display research progress
            st.markdown("## Research Progress")
            
            # Calculate overall progress
            total_sections = len(st.session_state.sections)
            completed_sections = len(st.session_state.section_data)
            current_section_idx = min(st.session_state.current_section, total_sections-1)
            
            # Progress bar
            progress_value = completed_sections / total_sections if total_sections > 0 else 0
            st.progress(progress_value)
            
            # Show progress status
            if st.session_state.research_phase == "researching":
                if current_section_idx < total_sections:
                    current_section = st.session_state.sections[current_section_idx]
                    st.markdown(f"""<div class='phase-indicator'>
                        Currently researching: <strong>{current_section['title']}</strong> (Section {current_section_idx + 1}/{total_sections})
                    </div>""", unsafe_allow_html=True)
            elif st.session_state.research_phase == "synthesizing":
                st.markdown("<div class='phase-indicator'>Synthesizing final report...</div>", unsafe_allow_html=True)
            elif st.session_state.research_phase == "complete":
                st.markdown("<div class='phase-indicator success-message'>Research complete! View the final report in the 'Final Report' tab.</div>", unsafe_allow_html=True)
            
            # Display timeline if we have at least one completed section
            if completed_sections > 0:
                st.markdown("### Research Timeline")
                timeline_data = create_research_timeline()
                timeline(timeline_data, height=400)
            
            # Display completed sections
            if completed_sections > 0:
                st.markdown("### Completed Sections")
                
                for section in st.session_state.sections:
                    section_id = section['id']
                    if section_id in st.session_state.section_data:
                        col1, col2, col3 = st.columns([5, 1, 1])
                        with col1:
                            st.markdown(f"""<div class='section-header'>{section['title']}</div>""", unsafe_allow_html=True)
                        with col2:
                            if st.button("View Content", key=f"view_{section_id}"):
                                toggle_section_display(section_id)
                        with col3:
                            if st.button("View Sources", key=f"sources_{section_id}"):
                                toggle_sources_display(section_id)
                                
                        # Display section content if expanded
                        if st.session_state.show_completed_section == section_id:
                            st.markdown(f"""<div class='info-box'>{st.session_state.section_data[section_id]}</div>""", unsafe_allow_html=True)
                            
                        # Display section sources if expanded
                        if st.session_state.show_section_sources == section_id:
                            sources = st.session_state.section_sources.get(section_id, [])
                            if sources:
                                with st.expander("Sources", expanded=True):
                                    for source in sources:
                                        st.markdown(f"[{source['title']}]({source['url']}) - {source['domain']}")
            
            # Start the research process if we're in the right phase and not yet complete
            if st.session_state.research_phase == "researching" and current_section_idx < total_sections:
                with st.spinner(f"Researching section: {st.session_state.sections[current_section_idx]['title']}"):
                    # Only execute this code if the tab is visible
                    if st.session_state.report_tab == "progress":
                        current_section = st.session_state.sections[current_section_idx]
                        section_result = research_section(current_section, query)
                        
                        # Store the section content
                        st.session_state.section_data[current_section['id']] = section_result["content"]
                        st.session_state.section_sources[current_section['id']] = section_result["sources"]
                        
                        # Move to the next section
                        st.session_state.current_section += 1
                        
                        if st.session_state.current_section >= total_sections:
                            st.session_state.research_phase = "synthesizing"
                        
                        # Rerun to update the UI
                        st.rerun()
            
            # Generate final report if in synthesis phase
            if st.session_state.research_phase == "synthesizing":
                with st.spinner("Generating final report..."):
                    final_report = synthesize_sections(
                        st.session_state.sections,
                        st.session_state.section_data,
                        query,
                        st.session_state.research_plan.get("introduction", ""),
                        st.session_state.research_plan.get("conclusion", "")
                    )
                    
                    # Store final report in session state
                    st.session_state.final_report = final_report
                    
                    # Generate timestamp for the file name
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.session_state.export_filename = f"research_report_{timestamp}.md"
                    
                    # Mark research as complete
                    st.session_state.research_phase = "complete"
                    st.session_state.report_tab = "report"  # Switch to report tab
                    
                    # Rerun to display the report
                    st.rerun()
    
    # PHASE 3: KNOWLEDGE GRAPH VISUALIZATION
    with graph_tab:
        st.markdown("## Knowledge Graph")
        
        # Only show if we have at least one completed section
        if len(st.session_state.section_data) > 0:
            # Attempt to generate a knowledge graph if it's empty
            if not st.session_state.knowledge_graph:
                with st.spinner("Generating knowledge graph..."):
                    # Combine all section content
                    all_content = "\n\n".join([content for content in st.session_state.section_data.values()])
                    
                    # Initialize a simple knowledge graph structure
                    initial_graph = {
                        "concepts": {},
                        "relationships": []
                    }
                    
                    # Update the knowledge graph
                    st.session_state.knowledge_graph = update_knowledge_graph(all_content, initial_graph)
            
            # Display the knowledge graph
            graph_data = render_knowledge_graph()
            if graph_data:
                st.image(f"data:image/png;base64,{graph_data}", use_container_width=True)
                
                # Display detailed concept information
                if 'concepts' in st.session_state.knowledge_graph:
                    with st.expander("Concept Details", expanded=False):
                        for concept, data in st.session_state.knowledge_graph.get('concepts', {}).items():
                            st.markdown(f"### {concept}")
                            st.markdown(f"**Confidence:** {data.get('confidence', 0):.2f}")
                            
                            if 'supporting_evidence' in data and data['supporting_evidence']:
                                st.markdown("**Supporting Evidence:**")
                                for evidence in data['supporting_evidence']:
                                    st.markdown(f"- {evidence}")
                            
                            if 'conflicting_evidence' in data and data['conflicting_evidence']:
                                st.markdown("**Conflicting Evidence:**")
                                for evidence in data['conflicting_evidence']:
                                    st.markdown(f"- {evidence}")
                            
                            st.markdown("---")
            else:
                st.info("Knowledge graph visualization will appear here as research progresses.")
        else:
            st.info("Knowledge graph will be generated once research begins.")
    
    # PHASE 4: FINAL REPORT
    with report_tab:
        if hasattr(st.session_state, 'final_report') and st.session_state.research_phase == "complete":
            # Display download button
            st.download_button(
                label="📥 Export Final Report",
                data=st.session_state.final_report,
                file_name=st.session_state.export_filename,
                mime="text/markdown"
            )
            
            # Display the final report
            st.markdown(st.session_state.final_report)
        else:
            st.info("The final report will appear here when research is complete.")
    
    # Track which tab is active for the research flow
    active_tabs = {"Research Plan": "plan", "Research Progress": "progress", 
                  "Knowledge Graph": "graph", "Final Report": "report"}
    
    # Update the active tab in session state
    for tab_name, tab_key in active_tabs.items():
        if plan_tab and tab_name in [t.label for t in [plan_tab, progress_tab, graph_tab, report_tab] if t.selected]:
            st.session_state.report_tab = tab_key 