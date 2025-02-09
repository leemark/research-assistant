# DEEP fREeSEARCH
An advanced AI-powered research assistant that performs comprehensive, multi-step analysis using Brave Search API and Google's Gemini 2.0 Flash Thinking model.

🔗 **Try it now:** [https://deepresearch.streamlit.app/](https://deepresearch.streamlit.app/)
📦 **GitHub Repository:** [https://github.com/leemark/research-assistant](https://github.com/leemark/research-assistant)

## Features
- 🤖 Multi-step reasoning with automated knowledge gap identification
- 📊 Dynamic knowledge graph construction and refinement
- 🔍 Intelligent web search powered by Brave Search API
- 🧠 Advanced analysis using Gemini 2.0 Flash Thinking model
- 🔄 Iterative research refinement (up to 5 iterations)
- 📝 Comprehensive research report generation
- 📥 Export functionality for markdown reports
- 🎯 Automatic query refinement and expansion
- 📈 Progress tracking and research status updates
- 🎨 Clean, user-friendly interface

## How It Works
1. The system starts with your research query
2. For each iteration:
   - Refines and expands the query
   - Performs targeted web searches
   - Analyzes findings using Gemini 2.0 Flash Thinking
   - Builds and updates knowledge graph
   - Identifies knowledge gaps
   - Determines if another iteration would be valuable
3. Synthesizes findings across all iterations
4. Generates a comprehensive research report

## Setup
1. Clone this repository
```bash
git clone https://github.com/leemark/research-assistant.git
cd research-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API keys:
   - Get a [Brave Search API key](https://api.search.brave.com/app/keys)
   - Get a [Google API key](https://makersuite.google.com/app/apikey) for Gemini Pro
   - Create `.streamlit/secrets.toml`:
   ```toml
   BRAVE_API_KEY = "your_brave_api_key_here"
   GOOGLE_API_KEY = "your_google_api_key_here"
   ```

## Usage
1. Run the app locally:
```bash
streamlit run research_app.py
```

2. Enter your research query
3. Set maximum iterations (1-5)
4. Review the progress as the system:
   - Refines your query
   - Performs multi-step research
   - Builds knowledge graph
   - Identifies and fills knowledge gaps
5. Get your comprehensive research report
6. Export the report in markdown format

## Report Structure
- Title and Timestamp
- Abstract
- Research Questions
- Executive Summary
- Key Findings and Analysis
- Implications and Applications
- Challenges and Limitations
- Future Directions
- Sources Analyzed

## Deployment
The app is deployed on Streamlit Cloud. You can:
- Use the live demo at [https://deepresearch.streamlit.app/](https://deepresearch.streamlit.app/)
- Deploy your own instance on Streamlit Cloud:
  1. Fork this repository
  2. Connect it to your Streamlit Cloud account
  3. Add your API keys in the Streamlit Cloud dashboard:
     - `BRAVE_API_KEY`
     - `GOOGLE_API_KEY`

## Contributing
Contributions are welcome! Areas for improvement include:
- Enhanced knowledge graph visualization
- Additional search sources
- Improved query refinement strategies
- Better source validation
- Custom iteration strategies

Please feel free to submit issues and pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details. 