# DEEP fREeSEARCH
An advanced AI-powered research assistant that performs comprehensive, section-based research with interactive visualization using Brave Search API and Google's Gemini 2.0 Flash Thinking model.

🔗 **Try it now:** [https://deepresearch.streamlit.app/](https://deepresearch.streamlit.app/)
📦 **GitHub Repository:** [https://github.com/leemark/research-assistant](https://github.com/leemark/research-assistant)

## Features
- 📋 **Smart Research Planning** - Automatically breaks topics into logical sections with targeted research questions
- 🔍 **Section-by-Section Research** - Conducts focused research on each section with proper source citation
- 🤖 **Advanced Analysis** - Powered by Gemini 2.0 Flash Thinking for in-depth content analysis
- 📊 **Knowledge Graph Visualization** - Interactive visualization of concept relationships and connections
- 🔄 **Research Timeline** - Visualizes the research progress with key milestones
- 🧩 **Interactive Section Review** - Expandable/collapsible sections with source viewing
- 📝 **Comprehensive Report Generation** - Well-structured reports with proper citations and organization
- 📥 **Export Functionality** - Download complete research reports in markdown format
- 🎯 **Human-in-the-Loop Feedback** - Review and approve research plans before execution
- 🎨 **Modern, Intuitive Interface** - Clean, tabbed design with visual progress indicators

## How It Works
1. **Planning Phase**
   - Generate a comprehensive research plan with sections
   - Review and approve the plan or request regeneration
   - Each section includes key questions to guide research

2. **Research Phase**
   - Conduct targeted research for each section independently
   - Generate section-specific search queries
   - Analyze content in the context of each section
   - Track sources and citations on a per-section basis

3. **Synthesis Phase**
   - Combine all section content into a cohesive report
   - Generate proper introduction and conclusion
   - Create table of contents with anchors
   - Consolidate sources with de-duplication

4. **Interactive Review**
   - Review completed sections while research is in progress
   - Explore the knowledge graph of concept relationships
   - View research timeline and progress indicators
   - Export the final report when complete

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

2. Enter your research query and set research depth
3. Review the generated research plan:
   - See how the topic is broken down into sections
   - Review key questions for each section
   - Approve the plan or request regeneration

4. Monitor the research progress:
   - Watch as each section is researched in sequence
   - Review completed sections while others are being researched
   - Explore sources for each section
   - View the knowledge graph as it develops

5. Get your comprehensive research report:
   - Review the final synthesized report
   - Export in markdown format for use in other applications

## Report Structure
- Title and Timestamp
- Table of Contents (with navigation anchors)
- Introduction
- Multiple Research Sections (as defined in the research plan)
- Conclusion
- Sources with Links

## Deployment
The app is deployed on Streamlit Cloud. You can:
- Use the live demo at [https://deepresearch.streamlit.app/](https://deepresearch.streamlit.app/)
- Deploy your own instance on Streamlit Cloud:
  1. Fork this repository
  2. Connect it to your Streamlit Cloud account
  3. Add your API keys in the Streamlit Cloud dashboard:
     - `BRAVE_API_KEY`
     - `GOOGLE_API_KEY`

## New in This Version
- **Research Planning Phase** - Smart topic breakdown into logical sections
- **Tabbed Interface** - Organize research workflow into logical tabs
- **Section-by-Section Research** - More focused, thorough analysis
- **Interactive Section Review** - Examine completed sections while others are being researched
- **Knowledge Graph Visualization** - See concept relationships and connections
- **Research Timeline** - Track research progress with timeline visualization
- **Improved UI/UX** - Better styling, progress indicators, and visual feedback
- **Enhanced Report Structure** - Better organization with proper sections and navigation

## Contributing
Contributions are welcome! Areas for improvement include:
- Additional search API integrations (Tavily, Perplexity, etc.)
- Support for other LLM providers (OpenAI, Anthropic, etc.)
- Enhanced knowledge graph interactions
- Custom report templates
- PDF export functionality
- Additional citation styles

Please feel free to submit issues and pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details. 