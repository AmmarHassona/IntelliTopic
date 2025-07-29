from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import openai
from dotenv import load_dotenv
import pandas as pd
import json
import re
from typing import List, Dict, Optional
import io
import tempfile
import httpx
from bs4 import BeautifulSoup
import asyncio
import aiohttp
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables
load_dotenv()

# Configure OpenAI
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Templates
templates = Jinja2Templates(directory="templates")

# Auto-clean existing topics on startup
@app.on_event("startup")
async def startup_event():
    """Clean existing topics and fix titles on application startup"""
    try:
        # Clean existing topics
        cleaned_count = clean_existing_topics()
        if cleaned_count > 0:
            print(f"✅ Auto-cleaned {cleaned_count} existing topics on startup")
        else:
            print("✅ No topics needed cleaning on startup")
        
        # Fix existing topic titles
        fixed_count = fix_existing_topic_titles()
        if fixed_count > 0:
            print(f"✅ Auto-fixed {fixed_count} existing topic titles on startup")
        else:
            print("✅ No topic titles needed fixing on startup")
            
    except Exception as e:
        print(f"⚠️ Error during startup cleanup: {e}")

# Custom Jinja2 filter to format topic content
def format_topic_content(content):
    """Format plain text topic content into HTML"""
    if not content:
        return "No content available"
    
    # First, clean any remaining HTML if present
    clean_content = re.sub(r'<[^>]+>', '', content)
    clean_content = re.sub(r'&amp;', '&', clean_content)
    clean_content = re.sub(r'&lt;', '<', clean_content)
    clean_content = re.sub(r'&gt;', '>', clean_content)
    
    # Remove extra whitespace and normalize
    clean_content = re.sub(r'\s+', ' ', clean_content).strip()
    
    # Define section patterns to look for
    section_patterns = [
        r'Scope Rating.*?\[(Easy|Medium|Hard)\]',
        r'Uniqueness Score.*?\[(\d+)\]',
        r'Brief Overview.*?\[(.+?)\]',
        r'Research Problem.*?\[(.+?)\]',
        r'Research Gap.*?\[(.+?)\]',
        r'Significance.*?\[(.+?)\]',
        r'Key Research Questions.*?\[(.+?)\]',
        r'Methodology.*?\[(.+?)\]',
        r'Expected Outcomes.*?\[(.+?)\]',
        r'Prerequisites.*?\[(.+?)\]',
        r'Innovation Factor.*?\[(.+?)\]',
        r'Project Description.*?\[(.+?)\]',
        r'Learning Outcomes.*?\[(.+?)\]',
        r'Project Components.*?\[(.+?)\]',
        r'Implementation Methods.*?\[(.+?)\]',
        r'Assessment Criteria.*?\[(.+?)\]',
        r'Target Students.*?\[(.+?)\]'
    ]
    
    # Extract sections
    sections = []
    remaining_content = clean_content
    
    for pattern in section_patterns:
        match = re.search(pattern, remaining_content, re.DOTALL | re.IGNORECASE)
        if match:
            # Find the section title
            section_start = match.start()
            section_end = match.end()
            
            # Extract the section title from the pattern
            if 'Scope Rating' in pattern:
                title = 'Scope Rating'
                value = match.group(1)
                content_text = f'<span class="scope-{value.lower()}">{value}</span>'
            elif 'Uniqueness Score' in pattern:
                title = 'Uniqueness Score'
                value = match.group(1)
                content_text = f'{value}/10'
            else:
                # Extract title from the pattern
                title = pattern.split('.*?')[0].replace('.*?', '').strip()
                value = match.group(1)
                content_text = value
            
            sections.append({
                'title': title,
                'content': content_text
            })
            
            # Remove this section from remaining content
            remaining_content = remaining_content[:section_start] + remaining_content[section_end:]
    
    # Format sections into HTML
    formatted_html = ""
    for section in sections:
        formatted_html += f'<div class="topic-section">'
        formatted_html += f'<div class="section-title">{section["title"]}</div>'
        formatted_html += f'<div class="section-content">{section["content"]}</div>'
        formatted_html += '</div>'
    
    # If no sections were found, format as a simple paragraph
    if not formatted_html:
        # Split by common delimiters and format
        parts = re.split(r'(Scope Rating|Uniqueness Score|Brief Overview|Research Problem|Research Gap|Significance|Key Research Questions|Methodology|Expected Outcomes|Prerequisites|Innovation Factor)', clean_content)
        
        if len(parts) > 1:
            for i in range(1, len(parts), 2):
                if i < len(parts) - 1:
                    title = parts[i].strip()
                    content = parts[i + 1].strip()
                    if content:
                        formatted_html += f'<div class="topic-section">'
                        formatted_html += f'<div class="section-title">{title}</div>'
                        formatted_html += f'<div class="section-content"><p>{content}</p></div>'
                        formatted_html += '</div>'
        else:
            formatted_html = f'<div class="topic-section"><div class="section-content"><p>{clean_content}</p></div></div>'
    
    return formatted_html

# Add the filter to templates
templates.env.filters["format_topic_content"] = format_topic_content

# User profiles storage (in production, use a proper database)
USER_PROFILES_FILE = "user_profiles.json"

def load_user_profiles() -> Dict:
    """Load user profiles from JSON file"""
    try:
        if os.path.exists(USER_PROFILES_FILE):
            with open(USER_PROFILES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading user profiles: {e}")
    return {}

def save_user_profiles(profiles: Dict):
    """Save user profiles to JSON file"""
    try:
        with open(USER_PROFILES_FILE, 'w', encoding='utf-8') as f:
            json.dump(profiles, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving user profiles: {e}")

def get_or_create_user_profile(user_id: str, role: str) -> Dict:
    """Get or create a user profile"""
    profiles = load_user_profiles()
    if user_id not in profiles:
        profiles[user_id] = {
            "role": role,
            "created_at": pd.Timestamp.now().isoformat(),
            "extracted_info": {
                "documents": [],
                "google_scholar": [],
                "user_inputs": []
            },
            "preferences": {},
            "statistics": {
                "topics_generated": 0,
                "documents_processed": 0,
                "scholar_profiles_analyzed": 0
            }
        }
        save_user_profiles(profiles)
    return profiles[user_id]

def update_user_profile(user_id: str, update_data: Dict):
    """Update user profile with new information"""
    profiles = load_user_profiles()
    if user_id in profiles:
        # Deep merge the update data
        for key, value in update_data.items():
            if key in profiles[user_id] and isinstance(profiles[user_id][key], dict):
                profiles[user_id][key].update(value)
            else:
                profiles[user_id][key] = value
        save_user_profiles(profiles)

async def scrape_google_scholar(profile_url: str) -> Dict:
    """Scrape Google Scholar profile for research information"""
    try:
        # Use a more robust approach with proper headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(profile_url, headers=headers)
            response.raise_for_status()
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract basic profile information
        profile_info = {
            "name": "",
            "affiliation": "",
            "research_interests": [],
            "publications": [],
            "citations": 0,
            "h_index": 0,
            "i10_index": 0
        }
        
        # Extract name (usually in the main heading)
        name_elem = soup.find('div', {'id': 'gsc_prf_in'})
        if name_elem:
            profile_info["name"] = name_elem.get_text().strip()
        
        # Extract affiliation
        affiliation_elem = soup.find('div', {'class': 'gsc_prf_il'})
        if affiliation_elem:
            profile_info["affiliation"] = affiliation_elem.get_text().strip()
        
        # Extract research interests
        interests_elem = soup.find('div', {'class': 'gsc_prf_il'})
        if interests_elem:
            interests = interests_elem.find_all('a')
            profile_info["research_interests"] = [interest.get_text().strip() for interest in interests]
        
        # Extract publication information
        publications = []
        pub_elements = soup.find_all('tr', {'class': 'gsc_a_tr'})
        for pub in pub_elements[:10]:  # Limit to first 10 publications
            title_elem = pub.find('a', {'class': 'gsc_a_at'})
            authors_elem = pub.find('div', {'class': 'gs_gray'})
            citations_elem = pub.find('a', {'class': 'gsc_a_ac'})
            
            if title_elem:
                publication = {
                    "title": title_elem.get_text().strip(),
                    "authors": authors_elem.get_text().strip() if authors_elem else "",
                    "citations": int(citations_elem.get_text().strip()) if citations_elem and citations_elem.get_text().strip().isdigit() else 0
                }
                publications.append(publication)
        
        profile_info["publications"] = publications
        
        # Extract citation metrics
        metrics = soup.find_all('td', {'class': 'gsc_rsb_std'})
        if len(metrics) >= 3:
            profile_info["citations"] = int(metrics[0].get_text().strip()) if metrics[0].get_text().strip().isdigit() else 0
            profile_info["h_index"] = int(metrics[1].get_text().strip()) if metrics[1].get_text().strip().isdigit() else 0
            profile_info["i10_index"] = int(metrics[2].get_text().strip()) if metrics[2].get_text().strip().isdigit() else 0
        
        return profile_info
        
    except Exception as e:
        return {"error": f"Failed to scrape Google Scholar profile: {str(e)}"}

async def extract_research_insights(profile_data: Dict) -> str:
    """Extract research insights from Google Scholar profile data for context understanding"""
    try:
        prompt = f"""Analyze the following Google Scholar profile data to understand the researcher's background and expertise (for context only):

Profile Information:
- Name: {profile_data.get('name', 'N/A')}
- Affiliation: {profile_data.get('affiliation', 'N/A')}
- Research Interests: {', '.join(profile_data.get('research_interests', []))}
- Total Citations: {profile_data.get('citations', 0)}
- H-index: {profile_data.get('h_index', 0)}
- i10-index: {profile_data.get('i10_index', 0)}

Recent Publications:
{chr(10).join([f"- {pub.get('title', 'N/A')} (Citations: {pub.get('citations', 0)})" for pub in profile_data.get('publications', [])])}

CONTEXT ANALYSIS (Do NOT suggest research topics based on existing work):

1. **Expertise Assessment:**
   - Core research areas and methodological strengths
   - Level of expertise and impact in their field
   - Technical skills and knowledge domains

2. **Research Patterns:**
   - Publication frequency and collaboration patterns
   - Citation impact and recognition in the field
   - Research evolution and trajectory

3. **Capability Profile:**
   - Teaching and mentoring experience indicators
   - Interdisciplinary connections and breadth
   - Innovation capacity and adaptability

4. **Field Context:**
   - Current position in their research domain
   - Network and collaboration potential
   - Emerging opportunities in their area

IMPORTANT: This analysis should inform understanding of the researcher's capabilities and background, NOT suggest topics similar to their existing work. The goal is to understand what they CAN do, not what they HAVE done.

Format the response as a structured background analysis suitable for informing NEW topic generation."""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.7,
        )
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error analyzing research profile: {str(e)}"

# Document parsing functions
async def parse_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return f"Error parsing PDF: {str(e)}"

async def parse_docx(file_content: bytes) -> str:
    """Extract text from DOCX file"""
    try:
        doc = Document(io.BytesIO(file_content))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        return f"Error parsing DOCX: {str(e)}"

async def parse_excel(file_content: bytes) -> str:
    """Extract text from Excel file"""
    try:
        df = pd.read_excel(io.BytesIO(file_content))
        return df.to_string()
    except Exception as e:
        return f"Error parsing Excel: {str(e)}"

async def parse_image(file_content: bytes) -> str:
    """Extract text from image using OCR"""
    try:
        image = Image.open(io.BytesIO(file_content))
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        return f"Error parsing image: {str(e)}"

async def parse_text_file(file_content: bytes) -> str:
    """Extract text from plain text file"""
    try:
        return file_content.decode('utf-8')
    except Exception as e:
        return f"Error parsing text file: {str(e)}"

async def parse_uploaded_files(files: List[UploadFile]) -> str:
    """Parse multiple uploaded files and return combined text"""
    if not files:
        return ""
    
    parsed_texts = []
    
    for file in files:
        if not file.filename:
            continue
            
        file_content = await file.read()
        file_extension = file.filename.lower().split('.')[-1]
        
        try:
            if file_extension == 'pdf':
                text = await parse_pdf(file_content)
            elif file_extension in ['docx', 'doc']:
                text = await parse_docx(file_content)
            elif file_extension in ['xlsx', 'xls']:
                text = await parse_excel(file_content)
            elif file_extension in ['png', 'jpg', 'jpeg', 'gif', 'bmp']:
                text = await parse_image(file_content)
            elif file_extension in ['txt', 'md']:
                text = await parse_text_file(file_content)
            else:
                text = f"Unsupported file type: {file_extension}"
            
            if text and not text.startswith("Error"):
                parsed_texts.append(f"--- Content from {file.filename} ---\n{text}\n")
                
        except Exception as e:
            parsed_texts.append(f"Error processing {file.filename}: {str(e)}")
    
    return "\n".join(parsed_texts)

async def fetch_relevant_papers(topic_title: str, max_papers: int = 5) -> List[Dict]:
    """Fetch relevant papers from arXiv API based on topic keywords"""
    try:
        # Extract key terms from topic title
        keywords = topic_title.replace(":", " ").replace("-", " ").split()
        # Filter out common words and keep meaningful terms
        meaningful_keywords = [kw for kw in keywords if len(kw) > 3 and kw.lower() not in ['the', 'and', 'for', 'with', 'using', 'based', 'analysis', 'study', 'research']]
        
        if not meaningful_keywords:
            return []
        
        # Use the most relevant keywords for search
        search_query = " ".join(meaningful_keywords[:3])
        encoded_query = quote_plus(search_query)
        
        # arXiv API endpoint
        url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={max_papers}&sortBy=relevance&sortOrder=descending"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    xml_content = await response.text()
                    return parse_arxiv_response(xml_content)
                else:
                    print(f"ArXiv API error: {response.status}")
                    return []
    except Exception as e:
        print(f"Error fetching papers: {e}")
        return []

def parse_arxiv_response(xml_content: str) -> List[Dict]:
    """Parse arXiv XML response and extract paper information"""
    try:
        root = ET.fromstring(xml_content)
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        
        papers = []
        entries = root.findall('.//atom:entry', namespace)
        
        for entry in entries:
            title_elem = entry.find('atom:title', namespace)
            summary_elem = entry.find('atom:summary', namespace)
            authors = entry.findall('.//atom:name', namespace)
            published_elem = entry.find('atom:published', namespace)
            id_elem = entry.find('atom:id', namespace)
            
            if title_elem is not None:
                title = title_elem.text.strip()
                summary = summary_elem.text.strip() if summary_elem is not None else ""
                author_names = [author.text for author in authors] if authors else []
                published = published_elem.text[:10] if published_elem is not None else ""
                paper_id = id_elem.text if id_elem is not None else ""
                
                papers.append({
                    'title': title,
                    'authors': author_names,
                    'summary': summary[:200] + "..." if len(summary) > 200 else summary,
                    'published': published,
                    'url': paper_id,
                    'arxiv_id': paper_id.split('/')[-1] if paper_id else ""
                })
        
        return papers
    except Exception as e:
        print(f"Error parsing arXiv response: {e}")
        return []

async def extract_paper_keywords(topic_content: str) -> List[str]:
    """Extract relevant keywords from topic content for paper search"""
    try:
        prompt = f"""Extract 3-5 most important technical keywords from this topic content for searching relevant academic papers. Return only the keywords separated by spaces, no explanations:

Topic Content:
{topic_content[:500]}

Keywords:"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.3,
        )
        
        keywords = response.choices[0].message.content.strip()
        return [kw.strip() for kw in keywords.split() if len(kw.strip()) > 2]
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []

def save_professor_topic(user_id: str, topic_content: str, mode: str) -> bool:
    """Save a topic to professor's profile"""
    try:
        profiles = load_user_profiles()
        profile = profiles.get(user_id, {})
        
        if profile.get("role") != "professor":
            return False
        
        # Extract topic information
        topic_info = extract_topic_info(topic_content, mode)
        
        # Initialize saved_topics if it doesn't exist
        if "saved_topics" not in profile:
            profile["saved_topics"] = []
        
        # Add the topic to saved topics
        profile["saved_topics"].append(topic_info)
        
        # Update the profile
        profiles[user_id] = profile
        save_user_profiles(profiles)
        
        return True
    except Exception as e:
        print(f"Error saving professor topic: {e}")
        return False

def extract_topic_info(topic_content: str, mode: str) -> Dict:
    """Extract structured information from topic content"""
    try:
        # Clean the HTML content to extract plain text
        import re
        
        # Remove HTML tags to get clean text
        clean_content = re.sub(r'<[^>]+>', '', topic_content)
        clean_content = re.sub(r'&amp;', '&', clean_content)
        clean_content = re.sub(r'&lt;', '<', clean_content)
        clean_content = re.sub(r'&gt;', '>', clean_content)
        
        # Extract title from the first line or use a default
        lines = clean_content.split('\n')
        title = "Untitled Topic"
        
        # Look for title in the content - find the first meaningful line
        for line in lines:
            line = line.strip()
            if line and not line.startswith('**') and not line.startswith('Scope') and not line.startswith('Uniqueness'):
                # Try to extract title from "Topic X: Title" or "Course Project X: Title" format
                title_match = re.search(r'(?:Topic|Course Project)\s+\d+:\s*(.+)', line)
                if title_match:
                    title = title_match.group(1).strip()
                    break
                elif len(line) > 10 and not line.startswith('**'):  # Use first substantial line as title
                    title = line
                    break
        
        # If we still have "Untitled Topic", try to extract from the brief overview
        if title == "Untitled Topic":
            overview_match = re.search(r'Brief Overview.*?\[(.+?)\]', clean_content, re.DOTALL)
            if overview_match:
                overview_text = overview_match.group(1).strip()
                # Take first sentence as title
                title = overview_text.split('.')[0] + '.'
        
        # If still no title, try to find any meaningful first line
        if title == "Untitled Topic":
            for line in lines:
                line = line.strip()
                if line and len(line) > 20 and not line.startswith('**') and not line.startswith('Scope') and not line.startswith('Uniqueness'):
                    title = line[:100] + '...' if len(line) > 100 else line
                    break
        
        # If still no title, try to extract from the content more aggressively
        if title == "Untitled Topic":
            # Look for any sentence that might be a title
            sentences = re.split(r'[.!?]', clean_content)
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence) > 30 and len(sentence) < 200:
                    # Check if it looks like a title (not metadata)
                    if not any(keyword in sentence.lower() for keyword in ['scope rating', 'uniqueness score', 'research problem', 'methodology']):
                        title = sentence + '.'
                        break
        
        # Extract keywords using OpenAI
        keywords = extract_keywords_from_content(clean_content)
        
        # Extract scope and uniqueness if present
        scope = "Medium"
        uniqueness = 8
        
        scope_match = re.search(r'Scope Rating.*?\[(Easy|Medium|Hard)\]', clean_content)
        if scope_match:
            scope = scope_match.group(1)
        
        uniqueness_match = re.search(r'Uniqueness Score.*?\[(\d+)\]', clean_content)
        if uniqueness_match:
            uniqueness = int(uniqueness_match.group(1))
        
        # Generate a unique ID using timestamp
        import time
        unique_id = f"topic_{int(time.time() * 1000)}"
        
        return {
            "id": unique_id,
            "content": clean_content,  # Store clean text content
            "title": title,
            "keywords": keywords,
            "scope": scope,
            "uniqueness": uniqueness,
            "created_at": pd.Timestamp.now().isoformat(),
            "mode": mode
        }
    except Exception as e:
        print(f"Error extracting topic info: {e}")
        return {
            "id": f"topic_{int(time.time() * 1000)}",
            "content": topic_content,
            "title": "Untitled Topic", 
            "scope": "Medium", 
            "uniqueness": 8, 
            "keywords": [], 
            "mode": mode
        }

def extract_keywords_from_content(content: str) -> List[str]:
    """Extract keywords from topic content using NLP techniques"""
    try:
        # Simple keyword extraction - can be enhanced with more sophisticated NLP
        words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
        
        # Filter out common words
        stop_words = {'this', 'that', 'with', 'from', 'they', 'have', 'will', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'would', 'there', 'could', 'other', 'than', 'first', 'very', 'after', 'where', 'most', 'over', 'think', 'also', 'around', 'another', 'into', 'during', 'before', 'while', 'under', 'never', 'become', 'himself', 'hundred', 'against', 'among', 'everything', 'through', 'within', 'further', 'himself', 'toward', 'together', 'however', 'neither', 'twenty', 'because', 'should', 'above', 'below', 'between', 'without', 'almost', 'sometimes', 'along', 'often', 'until', 'always', 'something', 'anything', 'nothing', 'everything', 'everyone', 'someone', 'anyone', 'nobody', 'somebody', 'anybody', 'everybody'}
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            if word not in stop_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:10]]
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []

def find_similar_topics(student_input: str, max_suggestions: int = 5) -> List[Dict]:
    """Find similar topics from professor profiles based on student input"""
    try:
        profiles = load_user_profiles()
        professor_topics = []
        
        print(f"DEBUG: Student input: {student_input}")
        print(f"DEBUG: Total profiles loaded: {len(profiles)}")
        
        # Collect all topics from ALL professors (not just the current user)
        for user_id, profile in profiles.items():
            if profile.get("role") == "professor" and "saved_topics" in profile:
                print(f"DEBUG: Professor {user_id} has {len(profile['saved_topics'])} saved topics")
                for topic in profile["saved_topics"]:
                    # Include both research and course project topics
                    topic_type = topic.get("mode", "research")
                    print(f"DEBUG: Found {topic_type} topic: {topic.get('title', 'No title')}")
                    professor_topics.append({
                        "topic": topic,
                        "professor_id": user_id,
                        "professor_name": profile.get("name", f"Professor {user_id}"),
                        "expertise": profile.get("extracted_info", {}).get("expertise", "General"),
                        "topic_type": topic_type
                    })
        
        print(f"DEBUG: Total professor topics found: {len(professor_topics)}")
        
        if not professor_topics:
            print("DEBUG: No professor topics found")
            return []
        
        # Create TF-IDF vectors for similarity comparison
        vectorizer = TfidfVectorizer(
            max_features=2000,  # Increased features
            stop_words='english',
            ngram_range=(1, 3),  # Include trigrams for better matching
            min_df=1,
            max_df=0.95  # Remove very common terms
        )
        
        # Prepare texts for comparison - include more comprehensive content
        texts = [student_input]
        topic_texts = []
        
        for pt in professor_topics:
            topic = pt['topic']
            # Include title, keywords, and content for better matching
            topic_text = f"{topic.get('title', '')} {' '.join(topic.get('keywords', []))} {topic.get('content', '')[:500]}"  # Include first 500 chars of content
            topic_texts.append(topic_text)
            texts.append(topic_text)
            print(f"DEBUG: Topic text for '{topic.get('title', '')}': {topic_text[:100]}...")
        
        # Create TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Calculate similarities
        student_vector = tfidf_matrix[0:1]
        topic_vectors = tfidf_matrix[1:]
        
        similarities = cosine_similarity(student_vector, topic_vectors).flatten()
        
        print(f"DEBUG: Similarity scores: {similarities}")
        
        # Sort by similarity and return top matches
        topic_similarities = list(zip(professor_topics, similarities))
        topic_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top suggestions with similarity scores - lower threshold
        suggestions = []
        for topic_info, similarity in topic_similarities[:max_suggestions]:
            if similarity > 0.01:  # Much lower threshold to catch more matches
                suggestions.append({
                    **topic_info,
                    "similarity_score": round(similarity * 100, 1)
                })
                print(f"DEBUG: Added suggestion '{topic_info['topic'].get('title', '')}' with score {similarity:.3f}")
        
        print(f"DEBUG: Final suggestions count: {len(suggestions)}")
        return suggestions
    except Exception as e:
        print(f"Error finding similar topics: {e}")
        import traceback
        traceback.print_exc()
        return []

def clean_existing_topics():
    """Clean up existing saved topics that contain HTML content"""
    try:
        profiles = load_user_profiles()
        cleaned_count = 0
        
        for user_id, profile in profiles.items():
            if profile.get("role") == "professor" and "saved_topics" in profile:
                for topic in profile["saved_topics"]:
                    if topic.get("content") and "<" in topic.get("content", ""):
                        # Clean the content
                        clean_content = re.sub(r'<[^>]+>', '', topic["content"])
                        clean_content = re.sub(r'&amp;', '&', clean_content)
                        clean_content = re.sub(r'&lt;', '<', clean_content)
                        clean_content = re.sub(r'&gt;', '>', clean_content)
                        clean_content = re.sub(r'\s+', ' ', clean_content).strip()
                        
                        # Extract clean title
                        title = "Untitled Topic"
                        lines = clean_content.split('\n')
                        for line in lines:
                            line = line.strip()
                            if line and not line.startswith('**') and not line.startswith('Scope') and not line.startswith('Uniqueness'):
                                title_match = re.search(r'(?:Topic|Course Project)\s+\d+:\s*(.+)', line)
                                if title_match:
                                    title = title_match.group(1).strip()
                                    break
                                elif len(line) > 10 and not line.startswith('**'):
                                    title = line
                                    break
                        
                        # Update the topic
                        topic["content"] = clean_content
                        topic["title"] = title
                        cleaned_count += 1
        
        if cleaned_count > 0:
            save_user_profiles(profiles)
            print(f"Cleaned {cleaned_count} existing topics")
        
        return cleaned_count
    except Exception as e:
        print(f"Error cleaning existing topics: {e}")
        return 0

def fix_existing_topic_titles():
    """Fix existing saved topics by extracting proper titles from their content"""
    try:
        profiles = load_user_profiles()
        fixed_count = 0
        
        for user_id, profile in profiles.items():
            if profile.get("role") == "professor" and "saved_topics" in profile:
                for topic in profile["saved_topics"]:
                    if topic.get("title") == "Untitled Topic" and topic.get("content"):
                        # Extract title from content using the same logic as extract_topic_info
                        clean_content = topic["content"]
                        
                        # Try to extract title from the content
                        title = "Untitled Topic"
                        
                        # Method 1: Look for "Brief Overview" section
                        overview_match = re.search(r'Brief Overview\s*(.+?)(?=Research Problem|Research Gap|Significance|Key Research Questions|Methodology|Expected Outcomes|Prerequisites|Innovation Factor|$)', clean_content, re.DOTALL | re.IGNORECASE)
                        if overview_match:
                            overview_text = overview_match.group(1).strip()
                            # Take first sentence as title
                            first_sentence = overview_text.split('.')[0].strip()
                            if first_sentence and len(first_sentence) > 10:
                                title = first_sentence + '.'
                        
                        # Method 2: If no overview, try to find any meaningful first line
                        if title == "Untitled Topic":
                            lines = clean_content.split('\n')
                            for line in lines:
                                line = line.strip()
                                if line and len(line) > 20 and not line.startswith('Scope') and not line.startswith('Uniqueness'):
                                    # Skip metadata lines
                                    if not any(keyword in line.lower() for keyword in ['scope rating', 'uniqueness score', 'research problem', 'methodology']):
                                        title = line[:100] + '...' if len(line) > 100 else line
                                        break
                        
                        # Method 3: Extract from any sentence that looks like a title
                        if title == "Untitled Topic":
                            sentences = re.split(r'[.!?]', clean_content)
                            for sentence in sentences:
                                sentence = sentence.strip()
                                if sentence and len(sentence) > 30 and len(sentence) < 200:
                                    # Check if it looks like a title (not metadata)
                                    if not any(keyword in sentence.lower() for keyword in ['scope rating', 'uniqueness score', 'research problem', 'methodology']):
                                        title = sentence + '.'
                                        break
                        
                        # Update the topic title
                        if title != "Untitled Topic":
                            topic["title"] = title
                            fixed_count += 1
                            print(f"Fixed title for topic {topic.get('id', 'unknown')}: {title[:50]}...")
        
        if fixed_count > 0:
            save_user_profiles(profiles)
            print(f"Fixed {fixed_count} existing topic titles")
        
        return fixed_count
    except Exception as e:
        print(f"Error fixing existing topic titles: {e}")
        return 0

@app.get("/", response_class=HTMLResponse)
async def form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/profile/{user_id}", response_class=HTMLResponse)
async def view_profile(request: Request, user_id: str):
    """View user profile with extracted information"""
    profiles = load_user_profiles()
    profile = profiles.get(user_id, {})
    
    return templates.TemplateResponse("profile.html", {
        "request": request,
        "profile": profile,
        "user_id": user_id
    })

@app.post("/save-topic")
async def save_topic(
    request: Request,
    user_id: str = Form(...),
    topic_content: str = Form(...),
    mode: str = Form(...)
):
    """Save a generated topic to professor's profile"""
    profiles = load_user_profiles()
    profile = profiles.get(user_id, {})
    
    if profile.get("role") != "professor":
        raise HTTPException(status_code=403, detail="Only professors can save topics")
    
    success = save_professor_topic(user_id, topic_content, mode)
    
    if success:
        return {"success": True, "message": "Topic saved successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to save topic")

@app.get("/topic-suggestions", response_class=HTMLResponse)
async def get_topic_suggestions(request: Request, user_id: str = None):
    """Get topic suggestions page"""
    return templates.TemplateResponse("suggestions.html", {
        "request": request,
        "user_id": user_id or "demo"
    })

@app.post("/find-suggestions", response_class=HTMLResponse)
async def find_suggestions(
    request: Request,
    user_id: str = Form(...),
    student_input: str = Form(...)
):
    """Find similar topics from professor profiles"""
    print(f"DEBUG: Received find-suggestions request")
    print(f"DEBUG: User ID: {user_id}")
    print(f"DEBUG: Student input: {student_input}")
    
    suggestions = find_similar_topics(student_input, max_suggestions=8)
    
    print(f"DEBUG: Found {len(suggestions)} suggestions")
    
    return templates.TemplateResponse("suggestions.html", {
        "request": request,
        "user_id": user_id,
        "student_input": student_input,
        "suggestions": suggestions,
        "found_suggestions": len(suggestions) > 0
    })

@app.post("/scrape-scholar", response_class=HTMLResponse)
async def scrape_scholar_profile(
    request: Request,
    user_id: str = Form(...),
    role: str = Form(...),
    scholar_url: str = Form(...)
):
    """Scrape Google Scholar profile and extract research insights"""
    
    # Validate role (only professors can access this)
    if role.lower() != "professor":
        raise HTTPException(status_code=403, detail="Only professors can access Google Scholar scraping")
    
    # Get or create user profile
    profile = get_or_create_user_profile(user_id, role)
    
    # Scrape Google Scholar profile
    profile_data = await scrape_google_scholar(scholar_url)
    
    if "error" in profile_data:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": profile_data["error"],
            "user_id": user_id,
            "role": role,
            "input_text": "",
            "course_description": "",
            "course_details": "",
            "mode": "research",
            "difficulty": "medium"
        })
    
    # Extract research insights using AI
    research_insights = await extract_research_insights(profile_data)
    
    # Update user profile
    scholar_entry = {
        "url": scholar_url,
        "scraped_at": pd.Timestamp.now().isoformat(),
        "profile_data": profile_data,
        "research_insights": research_insights
    }
    
    update_user_profile(user_id, {
        "extracted_info": {
            "google_scholar": profile.get("extracted_info", {}).get("google_scholar", []) + [scholar_entry]
        },
        "statistics": {
            "scholar_profiles_analyzed": profile.get("statistics", {}).get("scholar_profiles_analyzed", 0) + 1
        }
    })
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "scholar_result": research_insights,
        "scholar_profile": profile_data,
        "user_id": user_id,
        "role": role,
        "scholar_url": scholar_url,
        "input_text": "",
        "course_description": "",
        "course_details": "",
        "mode": "research",
        "difficulty": "medium"
    })

@app.post("/generate", response_class=HTMLResponse)
async def generate(
    request: Request,
    user_id: str = Form(...),
    role: str = Form(...),
    mode: str = Form(...),
    difficulty: str = Form(...),
    input_text: str = Form(...),
    course_description: str = Form(default=""),
    course_details: str = Form(default=""),
    files: List[UploadFile] = File(default=[])
):
    # Role-based access control
    if mode == "non-research" and role.lower() != "professor":
        raise HTTPException(status_code=403, detail="Only professors can access non-research mode")

    # Get or create user profile
    profile = get_or_create_user_profile(user_id, role)
    
    # Parse uploaded files if any
    file_content = ""
    uploaded_files_info = []
    if files and len(files) > 0:
        file_content = await parse_uploaded_files(files)
        
        # Store file information in profile
        for file in files:
            if file.filename:
                uploaded_files_info.append({
                    "filename": file.filename,
                    "uploaded_at": pd.Timestamp.now().isoformat(),
                    "size": len(await file.read()) if hasattr(file, 'read') else 0
                })
                # Reset file position for later use
                await file.seek(0)
    
    # Get user's extracted information for context
    extracted_context = ""
    if profile.get("extracted_info"):
        # Add Google Scholar insights
        for scholar_entry in profile["extracted_info"].get("google_scholar", []):
            extracted_context += f"\nGoogle Scholar Profile Analysis:\n{scholar_entry.get('research_insights', '')}\n"
        
        # Add previous document content
        for doc_entry in profile["extracted_info"].get("documents", []):
            extracted_context += f"\nPrevious Document: {doc_entry.get('filename', 'Unknown')}\n{doc_entry.get('content', '')}\n"
    
    # Build context based on mode - COMPLETELY SEPARATE for research vs course projects
    if mode == "research":
        # RESEARCH MODE: Include all context (background, files, Google Scholar, etc.)
        context = input_text
        if file_content:
            context += f"\n\nAdditional context from uploaded documents:\n{file_content}"
        if extracted_context:
            context += f"\n\nUser Profile Context:\n{extracted_context}"
        
        # Build the document content section
        document_section = ""
        if file_content:
            document_section = f"\nDocument Content:\n{file_content}\n"
            
    else:
        # COURSE PROJECT MODE: ONLY include course-specific context (NO research context)
        context = input_text  # User background/expertise
        if file_content:
            context += f"\n\nAdditional context from uploaded documents:\n{file_content}"
        
        # Add course-specific information
        if course_description:
            context += f"\n\nCourse Description:\n{course_description}"
        if course_details:
            context += f"\n\nAdditional Course Details:\n{course_details}"
        
        # Build the document content section for course projects
        document_section = ""
        if file_content:
            document_section = f"\nDocument Content:\n{file_content}\n"
        if course_description:
            document_section += f"\nCourse Description:\n{course_description}\n"
        if course_details:
            document_section += f"\nAdditional Course Details:\n{course_details}\n"

    if mode == "research":
        prompt = f"""Generate 3 COMPLETELY NEW and innovative thesis topics for a {role} using chain-of-thought reasoning.

DESIRED DIFFICULTY LEVEL: {difficulty.upper()}
- EASY: Suitable for beginners, basic concepts, fundamental research
- MEDIUM: Moderate complexity, some advanced concepts, intermediate research
- HARD: Advanced concepts, complex implementation, cutting-edge research
- MIXED: Variety of difficulty levels (1 easy, 1 medium, 1 hard)

CONTEXT INFORMATION (Use for understanding background, NOT for topic generation):
User Input: {input_text}{document_section}
User Profile Context: {extracted_context}

CHAIN-OF-THOUGHT PROCESS:
1. First, analyze the user's background and expertise from the context
2. Consider the desired difficulty level and adjust complexity accordingly
3. Identify emerging trends, gaps, and opportunities in their field
4. Consider interdisciplinary connections and novel applications
5. Think about future challenges and unexplored research directions
6. Generate topics that leverage their expertise but explore NEW territory

CRITICAL REQUIREMENTS:
- Topics MUST be completely new and not based on existing research from the context
- Use context only to understand the user's capabilities and field
- Focus on emerging areas, interdisciplinary approaches, or novel applications
- Each topic should represent a significant departure from current work
- Aim for high innovation and novelty scores (8-10)
- ADJUST SCOPE RATING based on the desired difficulty level

For each thesis topic, provide the following structured format:

## Topic [Number]: [Title]

**Scope Rating:** [Easy/Medium/Hard] - Rate based on complexity, time requirements, and resource needs (should match desired difficulty)
**Uniqueness Score:** [8-10] - Rate how novel and innovative this topic is (must be 8 or higher)
**Brief Overview:** [2-3 sentences summarizing the core concept]

**Research Problem:** [Detailed explanation of the research problem/objective]

**Research Gap:** [Describe the specific gap in current literature/knowledge this addresses]

**Significance & Impact:** [The potential impact and importance of this research]

**Key Research Questions:**
- [Question 1]
- [Question 2]
- [Question 3]

**Methodology:** [Suggested research approach and methods]

**Expected Outcomes:** [What contributions this research will make to the field]

**Prerequisites:** [Required background knowledge or skills]

**Innovation Factor:** [What makes this topic truly novel and different from existing research]

Remember: Use the context to understand the user's expertise and field, but generate topics that represent NEW research directions, not extensions of existing work. Ensure the scope rating aligns with the desired difficulty level."""
    else:
        prompt = f"""Generate 3 COMPLETELY NEW and innovative course project ideas for a {role} using chain-of-thought reasoning.

DESIRED DIFFICULTY LEVEL: {difficulty.upper()}
- EASY: Suitable for beginners, basic concepts, fundamental skills
- MEDIUM: Moderate complexity, some advanced concepts, intermediate skills
- HARD: Advanced concepts, complex implementation, expert-level skills
- MIXED: Variety of difficulty levels (1 easy, 1 medium, 1 hard)

CONTEXT INFORMATION (Use for understanding background, NOT for project generation):
User Input: {input_text}{document_section}

CHAIN-OF-THOUGHT PROCESS:
1. First, analyze the user's teaching background and expertise from the context
2. Consider the desired difficulty level and adjust project complexity accordingly
3. Identify emerging educational needs and gaps in current curricula
4. Consider interdisciplinary approaches and novel project-based learning methodologies
5. Think about future skills students will need and unexplored project areas
6. Generate course projects that leverage their expertise but explore NEW educational territory

CRITICAL REQUIREMENTS:
- Course projects MUST be completely new and not based on existing projects from the context
- Use context only to understand the user's teaching capabilities and field
- Focus on emerging subjects, interdisciplinary approaches, or novel project methodologies
- Each project should represent a significant departure from traditional assignments
- Aim for high innovation and novelty scores (8-10)
- ADJUST SCOPE RATING based on the desired difficulty level
- IGNORE any research-related context (Google Scholar, previous research papers, etc.)

For each course project, provide the following structured format:

## Course Project [Number]: [Title]

**Scope Rating:** [Easy/Medium/Hard] - Rate based on complexity, time requirements, and resource needs (should match desired difficulty)
**Uniqueness Score:** [8-10] - Rate how innovative and distinctive this project concept is (must be 8 or higher)
**Brief Overview:** [2-3 sentences summarizing the project concept]

**Project Description:** [Detailed description and learning objectives]

**Learning Outcomes:** [Specific skills and knowledge students will develop]

**Project Components:**
- [Component 1]
- [Component 2]
- [Component 3]
- [Component 4]

**Implementation Methods:** [Innovative project approaches and activities]

**Assessment Criteria:** [Evaluation methods and criteria]

**Prerequisites:** [Required background knowledge or skills]

**Target Students:** [Who would benefit most from this project]

**Innovation Factor:** [What makes this project truly novel and different from traditional assignments]

Remember: Use the context to understand the user's teaching expertise and field, but generate projects that represent NEW educational directions, not extensions of existing assignments. Focus ONLY on course-specific information and ignore any research context. Ensure the scope rating aligns with the desired difficulty level."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7,
        )
        generated = response.choices[0].message.content
    except Exception as e:
        generated = f"OpenAI API error: {str(e)}"

    # Fetch relevant papers for each topic
    relevant_papers = []
    if generated and "OpenAI API error" not in generated and mode == "research":
        try:
            # Extract topic titles from the generated content
            topic_sections = generated.split("## ")
            for section in topic_sections[1:]:  # Skip the first empty section
                lines = section.strip().split('\n')
                if lines:
                    topic_title = lines[0].strip()
                    # Fetch papers for this topic
                    papers = await fetch_relevant_papers(topic_title, max_papers=3)
                    if papers:
                        relevant_papers.extend(papers)
        except Exception as e:
            print(f"Error fetching papers: {e}")

    # Update user profile with new information
    update_data = {
        "statistics": {
            "topics_generated": profile.get("statistics", {}).get("topics_generated", 0) + 1
        },
        "extracted_info": {
            "user_inputs": profile.get("extracted_info", {}).get("user_inputs", []) + [{
                "input": input_text,
                "mode": mode,
                "timestamp": pd.Timestamp.now().isoformat()
            }]
        }
    }
    
    if uploaded_files_info:
        update_data["extracted_info"]["documents"] = profile.get("extracted_info", {}).get("documents", []) + uploaded_files_info
        update_data["statistics"]["documents_processed"] = profile.get("statistics", {}).get("documents_processed", 0) + len(uploaded_files_info)
    
    update_user_profile(user_id, update_data)

    # Get file names for display
    file_names = [file.filename for file in files] if files and len(files) > 0 else []

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": generated,
        "input_text": input_text,
        "course_description": course_description,
        "course_details": course_details,
        "user_id": user_id,
        "role": role,
        "mode": mode,
        "difficulty": difficulty,
        "uploaded_files": file_names,
        "file_content": file_content if file_content else "",
        "relevant_papers": relevant_papers[:10]  # Limit to 10 papers total
    })

@app.post("/enrich-topic", response_class=HTMLResponse)
async def enrich_topic(
    request: Request,
    topic_title: str = Form(...),
    topic_content: str = Form(...),
    role: str = Form(...),
    mode: str = Form(...),
    difficulty: str = Form(default="medium"),
    user_id: str = Form(...),
    course_description: str = Form(default=""),
    input_text: str = Form(default=""),
    course_details: str = Form(default="")
):
    """Enrich a specific topic with additional details"""
    
    if mode == "research":
        enrichment_prompt = f"""Enrich and expand the following thesis topic with additional details:

Topic: {topic_title}

Current Content:
{topic_content}

Please provide additional details for:

1. **Literature Review Strategy**: How to approach the literature review for this NEW topic
2. **Methodology Deep Dive**: Detailed research methods and data collection strategies
3. **Timeline & Milestones**: Suggested project timeline with key milestones
4. **Resource Requirements**: Equipment, software, funding, and other resources needed
5. **Potential Challenges**: Anticipated difficulties and mitigation strategies
6. **Collaboration Opportunities**: Potential collaborators or institutions for this innovative work
7. **Publication Strategy**: Target journals/conferences for this novel research
8. **Impact Assessment**: Broader implications and potential applications
9. **Innovation Validation**: How to validate the novelty and significance of this research
10. **Future Directions**: Potential extensions and follow-up research opportunities

Format the response with clear headings and bullet points. Focus on expanding the existing topic rather than suggesting similar research."""
    else:
        enrichment_prompt = f"""Enrich and expand the following course project idea with additional details:

Course Project: {topic_title}

Current Content:
{topic_content}

Please provide additional details for:

1. **Detailed Project Plan**: Step-by-step breakdown of project phases and activities
2. **Learning Resources**: Materials, tools, and references needed for the project
3. **Assessment Rubrics**: Detailed grading criteria and evaluation methods
4. **Technology Integration**: Digital tools and platforms to enhance project work
5. **Student Engagement Strategies**: Interactive project approaches and participation methods
6. **Differentiation Strategies**: Adaptations for diverse learning styles and skill levels
7. **Industry Connections**: Real-world applications, guest speakers, or industry partnerships
8. **Project Deliverables**: Specific outputs and presentations students will create
9. **Innovation Validation**: How to demonstrate the novelty and value of this project
10. **Future Adaptations**: How to evolve and improve this project over time
11. **Risk Management**: Potential challenges and mitigation strategies
12. **Success Metrics**: How to measure the effectiveness and impact of this project

Format the response with clear headings and bullet points. Focus on expanding the existing project rather than suggesting similar projects."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": enrichment_prompt}],
            max_tokens=1500,
            temperature=0.7,
        )
        enriched_content = response.choices[0].message.content
    except Exception as e:
        enriched_content = f"Error enriching topic: {str(e)}"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": enriched_content,
        "input_text": input_text,
        "course_description": course_description,
        "course_details": course_details,
        "user_id": user_id,
        "role": role,
        "mode": mode,
        "difficulty": difficulty,
        "uploaded_files": [],
        "file_content": "",
        "enriched_topic": topic_title
    })

@app.get("/debug-profiles")
async def debug_profiles():
    """Debug endpoint to check user profiles"""
    profiles = load_user_profiles()
    debug_info = {
        "total_profiles": len(profiles),
        "professors": [],
        "topics_count": 0
    }
    
    for user_id, profile in profiles.items():
        if profile.get("role") == "professor":
            topics_count = len(profile.get("saved_topics", []))
            debug_info["professors"].append({
                "user_id": user_id,
                "name": profile.get("name", "Unknown"),
                "topics_count": topics_count,
                "topics": [{"title": t.get("title", "No title"), "mode": t.get("mode", "Unknown")} for t in profile.get("saved_topics", [])]
            })
            debug_info["topics_count"] += topics_count
    
    return {"debug_info": debug_info}

@app.get("/clean-topics")
async def clean_topics():
    """Clean up existing saved topics that contain HTML content"""
    cleaned_count = clean_existing_topics()
    return {"message": f"Cleaned {cleaned_count} existing topics", "cleaned_count": cleaned_count}

@app.get("/fix-titles")
async def fix_titles_endpoint():
    """Manually trigger title fixing for existing topics"""
    try:
        fixed_count = fix_existing_topic_titles()
        return {"message": f"Fixed {fixed_count} topic titles", "fixed_count": fixed_count}
    except Exception as e:
        return {"error": str(e)}