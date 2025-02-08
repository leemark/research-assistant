# Deep Research Assistant

A Streamlit-powered research assistant that uses Brave Search API and Google's Gemini Pro model to perform comprehensive research and analysis.

## Features
- Web search using Brave Search API
- AI analysis using Google Gemini Pro
- Export functionality for research reports
- Clean, user-friendly interface

## Setup
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up your environment variables in Streamlit Cloud:
   - `BRAVE_API_KEY`
   - `GOOGLE_API_KEY`

## Local Development
1. Create a `.streamlit/secrets.toml` file with your API keys:
   ```toml
   BRAVE_API_KEY = "your_brave_api_key_here"
   GOOGLE_API_KEY = "your_google_api_key_here"
   ```
2. Run the app: `streamlit run research_app.py`

## Deployment
This app is deployed on Streamlit Cloud. Visit [your-app-url] to use it. 