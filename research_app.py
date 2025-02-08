import streamlit as st
import requests
import google.generativeai as genai
from typing import List, Dict
import os
from datetime import datetime

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
model = genai.GenerativeModel('gemini-pro')

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

def analyze_with_gemini(query: str, search_results: List[Dict]) -> str:
    """
    Analyze search results using Gemini
    """
    prompt = f"""
    Research Query: {query}
    
    Based on the following search results, provide a comprehensive analysis:
    
    {'-' * 50}
    """
    
    for idx, result in enumerate(search_results, 1):
        prompt += f"\n{idx}. Title: {result['title']}\nDescription: {result['description']}\nURL: {result['url']}\n"
    
    prompt += f"""
    {'-' * 50}
    
    Please provide:
    1. A comprehensive summary of the findings
    2. Key insights and patterns
    3. Different perspectives or conflicting information
    4. Potential gaps in the research
    5. Recommendations for further investigation
    
    Format your response in clear sections with markdown formatting.
    """
    
    response = model.generate_content(prompt)
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