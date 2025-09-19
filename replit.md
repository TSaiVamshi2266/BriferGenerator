# Overview

News Brief Generator is a Streamlit web application that transforms news articles into three distinct summary formats using the Groq API with intelligent selection based on keyword analysis. The application accepts both raw text and URL inputs, automatically extracts content from web pages, generates multiple summary styles (bullet points, academic abstract, and simple English), and uses NLTK-powered keyword overlap analysis to select the most relevant summary.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: Streamlit-based web interface providing a clean, interactive user experience
- **Input Handling**: Dual input mechanism supporting both raw text entry and URL-based article extraction
- **UI Components**: Expandable sections for summary comparisons, sidebar configuration for API key management, and score visualization

## Backend Architecture
- **Core Processing**: Object-oriented design with `NewsBriefGenerator` class encapsulating all summarization logic
- **Content Extraction**: Multi-tier approach using trafilatura as primary extractor with BeautifulSoup fallback for robust web scraping
- **Summary Generation**: Three distinct prompting strategies targeting different audience needs and communication styles
- **Selection Algorithm**: NLTK-powered keyword extraction with Jaccard similarity scoring for automated best-summary selection

## API Integration
- **AI Service**: Groq API integration using the `llama3-8b-8192` model for cost-effective, fast summarization
- **Concurrent Processing**: Asynchronous API calls for improved performance when generating multiple summaries
- **Error Handling**: Comprehensive error management for API failures, invalid URLs, and empty content scenarios

## Natural Language Processing
- **Text Processing**: NLTK toolkit for tokenization, stop word removal, and Porter stemming
- **Keyword Analysis**: Automated keyword extraction with frequency analysis and similarity scoring
- **Content Validation**: Input validation and content quality checks to ensure meaningful processing

## Configuration Management
- **API Key Storage**: Dual configuration approach supporting both Streamlit secrets (`.streamlit/secrets.toml`) and runtime UI input
- **Resource Caching**: Streamlit caching for NLTK data downloads to optimize initialization performance

# External Dependencies

## AI and Machine Learning Services
- **Groq API**: Primary AI service for text summarization using the llama3-8b-8192 model
- **NLTK**: Natural Language Toolkit for text processing, tokenization, and keyword analysis

## Web Content Processing
- **trafilatura**: Primary web content extraction library for clean article text parsing
- **BeautifulSoup4**: Secondary web scraping tool for fallback content extraction
- **requests**: HTTP client for web page fetching and URL validation

## Framework and UI
- **Streamlit**: Web application framework providing the interactive user interface
- **Python 3.x**: Runtime environment with standard library support for core functionality

## Configuration and Deployment
- **Streamlit Secrets**: Configuration management for secure API key storage
- **Replit**: Deployment platform with integrated package management and hosting