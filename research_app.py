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

# Configure page settings
st.set_page_config(
    page_title="DEEP fREeSEARCH",
    page_icon="🔍",
    layout="wide"
)

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

def update_knowledge_graph(current_findings: str, knowledge_graph: Dict) -> Dict:
    """
    Update the knowledge graph with new findings and connections.
    The knowledge graph tracks key concepts, their relationships, and confidence levels.
    """
    prompt = f"""
    You are a precise JSON generator. Analyze these findings and update the knowledge graph.
    
    Current Findings:
    {current_findings}

    Existing Knowledge Graph:
    {json.dumps(knowledge_graph, indent=2)}

    IMPORTANT: You must return a valid JSON object with exactly this structure:
    {{
        "concepts": {{
            "concept_name": {{
                "confidence": 0.95,
                "related_concepts": ["concept1", "concept2"],
                "supporting_evidence": ["evidence1", "evidence2"],
                "conflicting_evidence": ["conflict1", "conflict2"]
            }}
        }},
        "relationships": [
            {{
                "source": "concept1",
                "target": "concept2",
                "type": "relates_to",
                "confidence": 0.8
            }}
        ]
    }}

    Rules:
    1. All confidence values must be between 0.0 and 1.0
    2. All strings must be properly escaped
    3. Arrays must contain at least one item
    4. Concept names must be unique
    5. Return ONLY the JSON object, no other text
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
            st.warning("Failed to extract JSON from response. Using existing graph.")
            return knowledge_graph
            
        try:
            updated_graph = json.loads(json_str)
            
            # Validate the structure
            if not all(key in updated_graph for key in ['concepts', 'relationships']):
                st.warning("Invalid graph structure. Using existing graph.")
                return knowledge_graph
                
            # Validate confidence values
            for concept in updated_graph['concepts'].values():
                if not 0 <= concept['confidence'] <= 1:
                    st.warning("Invalid confidence values. Using existing graph.")
                    return knowledge_graph
                    
            for rel in updated_graph['relationships']:
                if not 0 <= rel['confidence'] <= 1:
                    st.warning("Invalid confidence values. Using existing graph.")
                    return knowledge_graph
                    
            return updated_graph
            
        except json.JSONDecodeError as e:
            st.warning(f"Failed to parse knowledge graph JSON: {str(e)}")
            return knowledge_graph
            
    except Exception as e:
        st.warning(f"Error updating knowledge graph: {str(e)}")
        return knowledge_graph

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
    """
    
    # Generate the section content
    chat = model.start_chat(history=[])
    with st.spinner(f"Analyzing sources for section: {section['title']}..."):
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

# Streamlit UI
st.title("DEEP fREeSEARCH")
st.markdown("*The AI-powered research assistant*")

# Input section
with st.form("research_form"):
    query = st.text_input("Enter your research query:", placeholder="What would you like to research?")
    max_iterations = st.slider("Maximum research iterations per section:", min_value=1, max_value=5, value=2)
    submitted = st.form_submit_button("Start Research")

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

if submitted and query:
    # Create containers for different phases
    plan_container = st.empty()
    progress_container = st.empty()
    status_container = st.empty()
    
    # PHASE 1: PLANNING
    if st.session_state.research_phase == "initial":
        with st.spinner("Generating research plan..."):
            research_plan = generate_research_plan(query)
            st.session_state.research_plan = research_plan
            st.session_state.sections = research_plan["sections"]
            st.session_state.research_phase = "planning"
    
    # Display the research plan for review
    if st.session_state.research_phase == "planning":
        plan_container.markdown("## Research Plan")
        plan_text = f"### {st.session_state.research_plan['title']}\n\n"
        plan_text += "#### Sections:\n"
        for section in st.session_state.research_plan["sections"]:
            plan_text += f"**{section['title']}**: {section['description']}\n\n"
            plan_text += "Key questions:\n"
            for q in section.get('key_questions', []):
                plan_text += f"- {q}\n"
            plan_text += "\n"
        
        plan_container.markdown(plan_text)
        
        # Buttons for plan feedback
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Proceed with this plan"):
                st.session_state.research_phase = "researching"
                plan_container.empty()
                st.rerun()
        with col2:
            if st.button("Regenerate plan"):
                st.session_state.research_plan = None
                st.session_state.research_phase = "initial"
                plan_container.empty()
                st.rerun()
    
    # PHASE 2: SECTION-BY-SECTION RESEARCH  
    if st.session_state.research_phase == "researching":
        # Calculate overall progress
        total_sections = len(st.session_state.sections)
        current_section_idx = st.session_state.current_section
        
        if current_section_idx < total_sections:
            # Update progress
            progress_container.progress((current_section_idx) / total_sections)
            current_section = st.session_state.sections[current_section_idx]
            status_container.markdown(f"**Researching Section {current_section_idx + 1}/{total_sections}: {current_section['title']}**")
            
            # Research the current section
            with st.spinner(f"Researching section: {current_section['title']}"):
                section_result = research_section(current_section, query)
                
                # Store the section content
                st.session_state.section_data[current_section['id']] = section_result["content"]
                st.session_state.section_sources[current_section['id']] = section_result["sources"]
                
                # Move to the next section
                st.session_state.current_section += 1
                
                if st.session_state.current_section >= total_sections:
                    st.session_state.research_phase = "synthesizing"
                
                # Rerun to show progress on next section
                st.rerun()
        
    # PHASE 3: SYNTHESIS
    if st.session_state.research_phase == "synthesizing":
        status_container.markdown("**Synthesizing final report...**")
        progress_container.progress(1.0)  # Show complete
        
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
            
            # Clear containers
            progress_container.empty()
            status_container.empty()
            
            # Rerun to display the report
            st.rerun()

# Display results if available
if hasattr(st.session_state, 'final_report') and st.session_state.research_phase == "complete":
    # Display download button in a small container at the top
    st.container().download_button(
        label="📥 Export Final Report",
        data=st.session_state.final_report,
        file_name=st.session_state.export_filename,
        mime="text/markdown"
    )
    
    # Display the final report in a dedicated section
    st.markdown(st.session_state.final_report) 