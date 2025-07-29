from fastapi import FastAPI, Form, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from openai import OpenAI
import os
import io
import json
import re
from typing import List, Dict, Optional
from dotenv import load_dotenv
import PyPDF2
from docx import Document
import pandas as pd
from PIL import Image
import pytesseract
import httpx
from bs4 import BeautifulSoup
import asyncio

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()
templates = Jinja2Templates(directory="templates")

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

@app.get("/", response_class=HTMLResponse)
async def form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/profile/{user_id}", response_class=HTMLResponse)
async def view_profile(request: Request, user_id: str):
    """View user profile and extracted information"""
    profiles = load_user_profiles()
    if user_id not in profiles:
        raise HTTPException(status_code=404, detail="User profile not found")
    
    profile = profiles[user_id]
    return templates.TemplateResponse("profile.html", {
        "request": request,
        "profile": profile,
        "user_id": user_id
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
            "role": role
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
        "scholar_url": scholar_url
    })

@app.post("/generate", response_class=HTMLResponse)
async def generate(
    request: Request,
    user_id: str = Form(...),
    role: str = Form(...),
    mode: str = Form(...),
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
    
    # Build context from user input, file content, and extracted information
    context = input_text
    if course_description:
        context += f"\n\nCourse Description:\n{course_description}"
    if course_details:
        context += f"\n\nAdditional Course Details:\n{course_details}"
    if file_content:
        context += f"\n\nAdditional context from uploaded documents:\n{file_content}"
    if extracted_context:
        context += f"\n\nUser Profile Context:\n{extracted_context}"

    # Build the document content section
    document_section = ""
    if file_content:
        document_section = f"\nDocument Content:\n{file_content}\n"
    if course_description:
        document_section += f"\nCourse Description:\n{course_description}\n"
    if course_details:
        document_section += f"\nAdditional Course Details:\n{course_details}\n"

    if mode == "research":
        prompt = f"""Generate 3 COMPLETELY NEW and innovative thesis topics for a {role} using chain-of-thought reasoning.

CONTEXT INFORMATION (Use for understanding background, NOT for topic generation):
User Input: {input_text}{document_section}
User Profile Context: {extracted_context}

CHAIN-OF-THOUGHT PROCESS:
1. First, analyze the user's background and expertise from the context
2. Identify emerging trends, gaps, and opportunities in their field
3. Consider interdisciplinary connections and novel applications
4. Think about future challenges and unexplored research directions
5. Generate topics that leverage their expertise but explore NEW territory

CRITICAL REQUIREMENTS:
- Topics MUST be completely new and not based on existing research from the context
- Use context only to understand the user's capabilities and field
- Focus on emerging areas, interdisciplinary approaches, or novel applications
- Each topic should represent a significant departure from current work
- Aim for high innovation and novelty scores (8-10)

For each thesis topic, provide the following structured format:

## Topic [Number]: [Title]

**Scope Rating:** [Easy/Medium/Hard] - Rate based on complexity, time requirements, and resource needs
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

Remember: Use the context to understand the user's expertise and field, but generate topics that represent NEW research directions, not extensions of existing work."""
    else:
        prompt = f"""Generate 3 COMPLETELY NEW and innovative course project ideas for a {role} using chain-of-thought reasoning.

CONTEXT INFORMATION (Use for understanding background, NOT for project generation):
User Input: {input_text}{document_section}
User Profile Context: {extracted_context}

CHAIN-OF-THOUGHT PROCESS:
1. First, analyze the user's teaching background and research expertise from the context
2. Identify emerging educational needs and gaps in current curricula
3. Consider interdisciplinary approaches and novel project-based learning methodologies
4. Think about future skills students will need and unexplored project areas
5. Generate course projects that leverage their expertise but explore NEW educational territory

CRITICAL REQUIREMENTS:
- Course projects MUST be completely new and not based on existing projects from the context
- Use context only to understand the user's teaching capabilities and field
- Focus on emerging subjects, interdisciplinary approaches, or novel project methodologies
- Each project should represent a significant departure from traditional assignments
- Aim for high innovation and novelty scores (8-10)

For each course project, provide the following structured format:

## Course Project [Number]: [Title]

**Scope Rating:** [Easy/Medium/Hard] - Rate based on complexity, time requirements, and resource needs
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

**Innovation Factor:** [What makes this project truly novel and different from existing assignments]

**Resource Requirements:** [Materials, tools, and resources needed]

**Timeline:** [Suggested project duration and milestones]

Remember: Use the context to understand the user's expertise and teaching style, but generate course projects that represent NEW educational directions, not extensions of existing assignments."""

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
        "uploaded_files": file_names,
        "file_content": file_content if file_content else ""
    })

@app.post("/enrich-topic", response_class=HTMLResponse)
async def enrich_topic(
    request: Request,
    topic_title: str = Form(...),
    topic_content: str = Form(...),
    role: str = Form(...),
    mode: str = Form(...),
    user_id: str = Form(...),
    course_description: str = Form(default="")
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
        "input_text": "",
        "course_description": course_description,
        "user_id": user_id,
        "role": role,
        "mode": mode,
        "uploaded_files": [],
        "file_content": "",
        "enriched_topic": topic_title
    })