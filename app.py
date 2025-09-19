import streamlit as st
import os
import re
import asyncio
from typing import Dict, List, Tuple, Optional
import requests
from bs4 import BeautifulSoup
import trafilatura
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
import groq

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data with error handling"""
    try:
        # Download both old and new punkt resources for compatibility
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        return True
    except Exception as e:
        st.error(f"Failed to download NLTK data: {str(e)}")
        return False

# Initialize NLTK data
download_nltk_data()

class NewsBriefGenerator:
    def __init__(self, api_key: str):
        """Initialize the News Brief Generator with Groq API key"""
        self.client = groq.Groq(api_key=api_key)
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def fetch_article_from_url(self, url: str) -> str:
        """
        Fetch and extract article text from URL using trafilatura
        Falls back to BeautifulSoup if trafilatura fails
        """
        try:
            # First try trafilatura for better content extraction
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded)
                if text and len(text.strip()) > 100:
                    return text.strip()
            
            # Fallback to BeautifulSoup
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            # Try to find main content
            content_selectors = [
                'article', 
                '[class*="content"]', 
                '[class*="article"]',
                '[class*="post"]',
                'main',
                '.entry-content',
                '#content'
            ]
            
            text = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    text = elements[0].get_text(separator=' ', strip=True)
                    break
            
            # If no specific content found, extract from all paragraphs
            if not text or len(text) < 100:
                paragraphs = soup.find_all('p')
                text = ' '.join([p.get_text(strip=True) for p in paragraphs])
            
            # Clean up the text
            text = re.sub(r'\s+', ' ', text).strip()
            
            if len(text) < 50:
                raise ValueError("Extracted text is too short")
                
            return text
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch URL: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to extract article text: {str(e)}")
    
    def generate_summary(self, style: str, article_text: str) -> str:
        """Generate summary using Groq API based on style"""
        prompts = {
            "bullet": f"Summarize the following news article in 5-7 concise bullet points, focusing on key events, people, and outcomes. Keep it under 250 words:\n\n{article_text}",
            "abstract": f"Write a 150-200 word abstract of this news article, capturing the essence, background, and implications in a formal, academic style:\n\n{article_text}",
            "simple": f"Rewrite this news article as a simple, easy-to-read summary in plain English (like for kids or non-native speakers), under 200 words:\n\n{article_text}"
        }
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompts[style]
                    }
                ],
                model="llama-3.1-8b-instant",
                max_tokens=300,
                temperature=0.3
            )
            
            content = chat_completion.choices[0].message.content
            return content.strip() if content else ""
            
        except Exception as e:
            raise Exception(f"Failed to generate {style} summary: {str(e)}")
    
    def extract_keywords(self, text: str, top_n: int = 20) -> List[str]:
        """Extract top keywords from text using NLTK"""
        try:
            # Tokenize and convert to lowercase
            tokens = word_tokenize(text.lower())
            
            # Remove punctuation, numbers, and stop words
            tokens = [
                self.stemmer.stem(word) 
                for word in tokens 
                if word.isalpha() and word not in self.stop_words and len(word) > 2
            ]
            
            # Get most common words
            word_freq = Counter(tokens)
            return [word for word, _ in word_freq.most_common(top_n)]
            
        except Exception as e:
            # Fallback to simple word extraction if NLTK fails
            st.warning(f"NLTK processing failed, using simple word extraction: {str(e)}")
            try:
                # Simple tokenization fallback
                import re
                words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
                words = [word for word in words if word not in {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}]
                word_freq = Counter(words)
                return [word for word, _ in word_freq.most_common(top_n)]
            except Exception as fallback_error:
                st.error(f"Both NLTK and fallback keyword extraction failed: {str(fallback_error)}")
                return []
    
    def compute_jaccard_similarity(self, set1: set, set2: set) -> float:
        """Compute Jaccard similarity between two sets"""
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def select_best_summary(self, summaries: Dict[str, str], original_text: str) -> Tuple[str, Dict[str, float], Dict[str, List[str]]]:
        """Select the best summary based on keyword overlap with original text"""
        original_keywords = set(self.extract_keywords(original_text))
        
        scores = {}
        all_keywords = {"original": list(original_keywords)}
        
        for style, summary in summaries.items():
            summary_keywords = set(self.extract_keywords(summary))
            all_keywords[style] = list(summary_keywords)
            scores[style] = self.compute_jaccard_similarity(original_keywords, summary_keywords)
        
        # Select best summary (bullet-point as default in case of tie)
        best_style = max(scores.keys(), key=lambda x: (scores[x], x == "bullet"))
        
        return best_style, scores, all_keywords

def main():
    st.set_page_config(
        page_title="News Brief Generator",
        page_icon="📰",
        layout="wide"
    )
    
    st.title("📰 News Brief Generator")
    st.markdown("Generate and compare different summary styles of news articles using AI")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # API Key handling - secure method
    # First try to get from environment/secrets (deployed apps)
    api_key = os.getenv("GROQ_API_KEY", "")
    
    # Only show input field if no environment key is available
    if not api_key:
        api_key = st.sidebar.text_input(
            "Groq API Key",
            type="password",
            placeholder="Enter your Groq API key",
            help="Enter your Groq API key or set it in .streamlit/secrets.toml"
        )
        
        if not api_key:
            st.sidebar.warning("⚠️ Please enter your Groq API key to continue")
            st.sidebar.info("💡 For production, store the key in `.streamlit/secrets.toml`")
            return
    else:
        # Key is loaded from environment - show success message
        st.sidebar.success("✅ API Key loaded securely from environment")
        st.sidebar.info("🔒 Your API key is secure and not visible")
    
    # Initialize generator
    try:
        generator = NewsBriefGenerator(api_key)
    except Exception as e:
        st.error(f"Failed to initialize generator: {str(e)}")
        return
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Input")
        
        # Input options
        input_method = st.radio("Choose input method:", ["Text Input", "URL Input"])
        
        article_text = ""
        
        if input_method == "Text Input":
            # Sample article for testing
            sample_article = """
            Tech giant Apple announced today the launch of its revolutionary new iPhone 15 series, featuring advanced AI capabilities and improved battery life. The new devices, unveiled at the company's annual September event, include four models ranging from $799 to $1,199.

            CEO Tim Cook highlighted the integration of artificial intelligence features that can automatically organize photos, suggest responses to messages, and optimize device performance. The flagship iPhone 15 Pro Max boasts a 48-megapixel camera system and promises up to 29 hours of video playback.

            Industry analysts predict strong sales for the holiday season, with pre-orders beginning this Friday. The phones will be available in stores starting September 22nd. Apple's stock price surged 3.2% following the announcement, reaching a new all-time high of $189.50 per share.

            Environmental initiatives were also emphasized, with Apple claiming the new devices are made from 100% recycled rare earth elements. The company continues its commitment to becoming carbon neutral across its entire supply chain by 2030.
            """
            
            article_text = st.text_area(
                "Article Text",
                value=sample_article,
                height=300,
                help="Paste the news article text here"
            )
            
        else:  # URL Input
            url = st.text_input(
                "Article URL",
                placeholder="https://example.com/news-article",
                help="Enter the URL of the news article"
            )
            
            if url and st.button("Fetch Article"):
                with st.spinner("Fetching article..."):
                    try:
                        article_text = generator.fetch_article_from_url(url)
                        st.success("✅ Article fetched successfully!")
                        st.text_area("Fetched Article Preview", value=article_text[:500] + "...", height=150, disabled=True)
                    except Exception as e:
                        st.error(f"❌ {str(e)}")
                        article_text = ""
        
        # Control buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            generate_clicked = st.button("🚀 Generate Briefs", type="primary", disabled=not article_text.strip())
        with col_btn2:
            if st.button("🗑️ Clear"):
                st.rerun()
    
    with col2:
        st.header("Summary Scores")
        
        # Placeholder for scores table
        scores_placeholder = st.empty()
    
    # Generate summaries
    if generate_clicked and article_text.strip():
        with st.spinner("Generating summaries..."):
            try:
                # Generate all three summaries
                summaries = {}
                progress_bar = st.progress(0)
                
                styles = ["bullet", "abstract", "simple"]
                for i, style in enumerate(styles):
                    summaries[style] = generator.generate_summary(style, article_text)
                    progress_bar.progress((i + 1) / len(styles))
                
                progress_bar.empty()
                
                # Select best summary
                best_style, scores, keywords = generator.select_best_summary(summaries, article_text)
                
                # Display scores in sidebar
                with col2:
                    score_data = {
                        "Summary Type": ["Bullet Points", "Abstract", "Simple English"],
                        "Overlap Score": [f"{scores['bullet']:.3f}", f"{scores['abstract']:.3f}", f"{scores['simple']:.3f}"],
                        "Best": ["✅" if best_style == "bullet" else "", 
                                "✅" if best_style == "abstract" else "", 
                                "✅" if best_style == "simple" else ""]
                    }
                    st.dataframe(score_data, hide_index=True)
                
                # Display results
                st.header("Generated Summaries")
                
                # Show best summary first
                style_names = {
                    "bullet": "🔸 Bullet Points Summary",
                    "abstract": "📄 Abstract Summary", 
                    "simple": "👶 Simple English Summary"
                }
                
                if best_style == "bullet":
                    st.success(f"**Best Summary: {style_names[best_style]}** (Score: {scores[best_style]:.3f})")
                    st.markdown(summaries[best_style])
                    
                    with st.expander(f"{style_names['abstract']} (Score: {scores['abstract']:.3f})"):
                        st.markdown(summaries['abstract'])
                    
                    with st.expander(f"{style_names['simple']} (Score: {scores['simple']:.3f})"):
                        st.markdown(summaries['simple'])
                        
                elif best_style == "abstract":
                    st.success(f"**Best Summary: {style_names[best_style]}** (Score: {scores[best_style]:.3f})")
                    st.markdown(summaries[best_style])
                    
                    with st.expander(f"{style_names['bullet']} (Score: {scores['bullet']:.3f})"):
                        st.markdown(summaries['bullet'])
                    
                    with st.expander(f"{style_names['simple']} (Score: {scores['simple']:.3f})"):
                        st.markdown(summaries['simple'])
                        
                else:  # simple
                    st.success(f"**Best Summary: {style_names[best_style]}** (Score: {scores[best_style]:.3f})")
                    st.markdown(summaries[best_style])
                    
                    with st.expander(f"{style_names['bullet']} (Score: {scores['bullet']:.3f})"):
                        st.markdown(summaries['bullet'])
                    
                    with st.expander(f"{style_names['abstract']} (Score: {scores['abstract']:.3f})"):
                        st.markdown(summaries['abstract'])
                
                # Show keyword analysis
                with st.expander("🔍 Keyword Analysis"):
                    st.subheader("Top Keywords Comparison")
                    
                    col_orig, col_best = st.columns(2)
                    with col_orig:
                        st.write("**Original Article Keywords:**")
                        st.write(", ".join(keywords["original"][:10]))
                    
                    with col_best:
                        st.write(f"**Best Summary ({best_style.title()}) Keywords:**")
                        st.write(", ".join(keywords[best_style][:10]))
                
            except Exception as e:
                st.error(f"❌ Error generating summaries: {str(e)}")

if __name__ == "__main__":
    main()
