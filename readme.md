# GrantScope: AI-Powered Grant Data Exploration Dashboard

## Overview
GrantScope is an advanced, AI-driven interactive tool designed to revolutionize how grant analysts, grant writers, and researchers explore and understand complex grant data. By leveraging cutting-edge AI technology and data visualization techniques, GrantScope provides deep, context-aware insights tailored to specific project themes and goals, making it easier than ever to identify funding opportunities, understand funding trends, and enhance grant writing strategies.

## Key Features

### AI-Powered Comprehensive Analysis
- **Project Theme Integration**: Analyzes all data in the context of your specific project theme and goals.
- **Full Report Generation**: Creates a detailed, AI-generated report covering all aspects of the grant data, considering your project's unique focus.
- **Dynamic Insights**: Provides real-time, AI-generated insights for each visualization and analysis section.

### Interactive Data Visualizations
- **Data Summary Dashboard**: Offers an AI-enhanced overview of key metrics, top funders, and grant distributions.
- **Grant Amount Analysis**:
  - Distribution charts across USD clusters
  - Time-based scatter plots with trend analysis
  - Heatmaps for geographical and subject area insights
- **Word Clouds**: AI-analyzed visualizations of grant description themes.
- **Treemaps**: Hierarchical exploration of grant allocations by subject, population, and strategy.

### Detailed Analysis Tools
- **Relationship Analysis**: AI-driven exploration of connections between funders, recipients, and grant characteristics.
- **Top Categories Analysis**: In-depth look at the most significant categories by unique grant count.
- **Grant Description Deep Dive**: Advanced text analysis with AI-generated insights on themes and patterns.

### User-Centric Design
- **Customized User Roles**: Tailored experiences for grant analysts/writers and general users.
- **AI-Assisted Exploration**: Users can ask questions in natural language about any aspect of the data.
- **Downloadable Reports**: Generate and download comprehensive AI-analyzed reports and data extracts.

## How to Use GrantScope

1. **Project Setup**:
   - Upload your grant data file or use the preloaded dataset.
   - Enter your project theme and objectives to tailor the AI analysis.

2. **Explore the Dashboard**:
   - Navigate through different analysis sections.
   - Interact with visualizations to filter and focus on specific aspects.

3. **AI-Powered Insights**:
   - Read AI-generated analyses for each chart and section.
   - Ask custom questions to dive deeper into specific areas of interest.

4. **Generate Comprehensive Report**:
   - Use the "Generate Full Report" feature for a complete AI analysis of your data in the context of your project.

5. **Download and Share**:
   - Export visualizations, data extracts, and the full AI-generated report for further use.

## Technology Stack
- **Frontend**: Streamlit for the interactive web application
- **Data Processing**: Pandas and NumPy
- **Visualization**: Plotly and Matplotlib
- **AI and NLP**: OpenAI's GPT-4, LlamaIndex
- **Data Source**: Candid API for grant data

## Getting Started

### Web Version
Access GrantScope at [grantscope.streamlit.app](https://grantscope.streamlit.app).

### Local Installation
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/GrantScope.git
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your OpenAI API key as an environment variable or through the UI.
4. Run the application:
   ```
   streamlit run app.py
   ```

## Contributing
We welcome contributions to enhance GrantScope! If you have ideas, suggestions, or bug reports, please open an issue or submit a pull request. Let's collaborate to make this tool even more powerful for the grant analysis community.

## License
This project is licensed under the GNU General Public License v3.0.

---

GrantScope is developed by [Christopher Collins](https://www.linkedin.com/in/cctopher/), leveraging advanced AI models and data visualization techniques to provide dynamic, context-aware insights into the world of grants. We hope this tool empowers you to uncover valuable insights, identify potential funding opportunities, and craft compelling grant proposals tailored to your unique projects. Happy exploring and grant writing!
