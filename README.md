# News Brief Generator

A Streamlit web application that generates three different summary styles from news articles using the Groq API and selects the best one based on keyword overlap analysis.

## Features

- **Multi-Input Support**: Accept both raw text and URL inputs
- **Smart Content Extraction**: Automatically extract article content from URLs using trafilatura and BeautifulSoup
- **Three Summary Styles**:
  - Bullet Points: 5-7 concise key points
  - Abstract: 150-200 word formal academic summary
  - Simple English: Easy-to-read summary for general audiences
- **Intelligent Selection**: Uses NLTK for keyword extraction and Jaccard similarity to select the best summary
- **Interactive UI**: Clean Streamlit interface with expandable sections and score comparisons
- **Error Handling**: Comprehensive error handling for invalid URLs, API failures, and empty inputs

## Setup Instructions

### 1. Install Dependencies

The following packages are required:
- streamlit
- groq
- requests
- beautifulsoup4
- trafilatura
- nltk

### 2. Configure API Key

You have two options for setting up your Groq API key:

**Option A: Using secrets.toml (Recommended for production)**
1. Get your API key from [Groq Console](https://console.groq.com/)
2. Edit `.streamlit/secrets.toml` and replace `your-groq-api-key-here` with your actual API key

**Option B: Using the UI**
1. Enter your API key in the sidebar when running the application

### 3. Run the Application

```bash
streamlit run app.py --server.port 5000
