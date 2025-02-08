# Deep Research Assistant

A Streamlit-powered research assistant that uses Brave Search API and Google's Gemini Pro model to perform comprehensive research and analysis.

## Features
- Intelligent web search powered by Brave Search API
- Advanced AI analysis using Google Gemini Pro
- Multi-source research synthesis
- Customizable search parameters
- Export functionality for research reports
- Interactive research workspace
- Clean, user-friendly interface
- Real-time analysis and summarization

## Setup
1. Clone this repository
```bash
git clone https://github.com/yourusername/deep-research-assistant.git
cd deep-research-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
   - For local development, create `.streamlit/secrets.toml`:
   ```toml
   BRAVE_API_KEY = "your_brave_api_key_here"
   GOOGLE_API_KEY = "your_google_api_key_here"
   ```
   - For Streamlit Cloud deployment, add these variables in the Streamlit dashboard:
     - `BRAVE_API_KEY`
     - `GOOGLE_API_KEY`

## Local Development
1. Ensure your API keys are set in `.streamlit/secrets.toml`
2. Run the app:
```bash
streamlit run research_app.py
```

## Usage
1. Enter your research topic or question
2. Adjust search parameters if needed
3. Review the search results and AI analysis
4. Export your research findings

## Deployment
The app is deployed on Streamlit Cloud.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details. 