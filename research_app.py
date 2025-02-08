import streamlit as st
import requests
import google.generativeai as genai
from typing import List, Dict, Tuple
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
if 'refined_query' not in st.session_state:
    st.session_state.refined_query = ""

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

def write_final_report(refined_query: str, analyses: List[Tuple[str, str]], search_results: List[Dict], initial_report: str) -> str:
    """
    Generate a final combined research report using the initial analysis as context.
    
    The final report should include:
    - Title of Paper
    - Generated on: <timestamp>
    - Table of Contents
    - Abstract
    - Research Questions
    - Executive Summary (combined across all analyses)
    - Analyses 1-5
    - Sources
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Build the final report prompt without sources
    prompt = f"""
    Using the initial report context provided below, draft a final combined research report with the following structure:

    [Your Title Here, e.g., "Deep Research on {refined_query}"]
    Generated on: {current_time}

    ## Table of Contents
    1. [Table of Contents](#table-of-contents)
    2. [Abstract](#abstract)
    3. [Research Questions](#research-questions)
    4. [Executive Summary](#executive-summary)
    5. [Title of Analysis 1](#title-of-analysis-1)
    6. [Title of Analysis 2](#title-of-analysis-2)
    7. [Title of Analysis 3](#title-of-analysis-3)
    8. [Title of Analysis 4](#title-of-analysis-4)
    9. [Title of Analysis 5](#title-of-analysis-5)
    10. [Sources Analyzed](#sources-analyzed)

    ## Abstract
    Provide a brief summary of the entire research report.

    ## Research Questions
    {refined_query}

    ## Executive Summary
    Produce one combined executive summary for the entire report using 5-8 bullet points or 3-4 paragraphs.
    Ensure that it captures the most critical findings and recommendations across all analyses.

    ## Analyses
    {initial_report}
    """
    
    # Start a new Gemini chat and get the main report
    chat = model.start_chat(history=[])
    with st.spinner("Generating final report..."):
         final_response = chat.send_message(prompt)
         main_report = final_response.text

    # Add sources section separately
    sources_section = "\n## Sources Analyzed\n"
    for idx, result in enumerate(search_results, 1):
        sources_section += f"\n{idx}. [{result['title']}]({result['url']})"
    
    # Combine main report with sources
    complete_report = f"{main_report.strip()}\n{sources_section}"
    
    return complete_report

# Streamlit UI
st.title("🔍 Deep Research Assistant")
st.markdown("Powered by Google Gemini 2.0 Flash Thinking and Brave Search")

# Input section
with st.form("research_form"):
    query = st.text_input("Enter your research query:", placeholder="What would you like to research?")
    submitted = st.form_submit_button("Start Research")

if submitted and query:
    # Create a container for initial outputs that we'll clear later
    initial_output_container = st.empty()
    
    with initial_output_container:
        with st.spinner("Refining research query..."):
            refined_query = refine_research_query(query)
        st.session_state.refined_query = refined_query  # Store in session state
        st.markdown(f"**Refined Research Query:** {refined_query}")

        with st.spinner("Generating search queries..."):
             web_search_queries = simplify_search_query(refined_query)
    
    all_search_results = []
    all_analyses = []

    for i, web_search_query in enumerate(web_search_queries, 1):
        with st.spinner(f"Processing search query {i} of 5: {web_search_query}"):
            # Search using current query
            current_results = brave_search(web_search_query)
            all_search_results.extend(current_results)
            
            if current_results:
                # Analyze current results
                current_analysis = analyze_with_gemini(f"{refined_query} (Search Query {i}: {web_search_query})", current_results)
                all_analyses.append((web_search_query, current_analysis))

    # Store results in session state
    st.session_state.search_results = all_search_results
    st.session_state.analyses = all_analyses
    
    # Generate final report automatically
    initial_report = ""
    for idx, (search_query, analysis) in enumerate(st.session_state.analyses, 1):
        analysis_lines = analysis.split('\n')
        title = None
        for line in analysis_lines:
            if line.strip().startswith('Research Report:') or line.strip().startswith('Title:'):
                title = line.split(':', 1)[1].strip()
                break
        
        if not title:
            title = to_headline_case(search_query)
            
        initial_report += f"## {title}\n{analysis}\n\n"

    final_report = write_final_report(
        refined_query=st.session_state.refined_query,
        analyses=st.session_state.analyses,
        search_results=st.session_state.search_results,
        initial_report=initial_report
    )
    
    # Clean up the report
    final_report = final_report.replace("```markdown", "").replace("```", "").strip()
    
    # Store final report in session state
    st.session_state.final_report = final_report
    
    # Generate timestamp for the file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.export_filename = f"research_report_{timestamp}.md"
    
    # Clear the initial output container once we have the final report
    initial_output_container.empty()

# Display results if available
if hasattr(st.session_state, 'final_report'):
    # Display download button in a small container at the top
    st.container().download_button(
        label="📥 Export Final Report",
        data=st.session_state.final_report,
        file_name=st.session_state.export_filename,
        mime="text/markdown"
    )
    
    # Display the final report in a dedicated section
    st.markdown("## 📊 Final Research Report")
    st.markdown(st.session_state.final_report) 