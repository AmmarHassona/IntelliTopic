# IntelliTopic - AI-Powered Academic Idea Generator

IntelliTopic is a sophisticated web application that generates personalized academic ideas for students and professors using AI. The application supports multi-modality input, allowing users to upload documents that are parsed and analyzed to create more relevant and personalized suggestions.

## 📸 Demo

[![Watch the Demo](https://github.com/user-attachments/assets/d8b5939d-0cc4-4c7f-971e-e663fb00a67f)](https://drive.google.com/file/d/1w8T0Kvf028E3JHLRNxeXj4vdoXnAm3zy/view?usp=drive_link)

---

## 🌟 Features

### Core Functionality
- **Dual User Types**: Support for both students and professors
- **Two Generation Modes**:
  - **Research Mode**: Generate comprehensive thesis topics
  - **Non-Research Mode**: Generate creative course projects
- **AI-Powered Suggestions**: Uses OpenAI GPT-3.5-turbo for intelligent idea generation

### Multi-Modality Support
- **Document Upload**: Upload multiple file types for enhanced personalization
- **Supported File Formats**:
  - **PDF Documents**: Research papers, CVs, academic documents
  - **Word Documents**: Course materials, syllabi, proposals
  - **Excel Files**: Data sets, course schedules, research data
  - **Images**: Screenshots, diagrams, handwritten notes (OCR support)
  - **Text Files**: Notes, outlines, plain text documents

### Advanced Document Processing
- **T<img width="1915" height="897" alt="Screenshot 2025-08-13 163926" src="https://github.com/user-attachments/assets/c4c32205-e514-406a-9f02-734588d83e47" />
ext Extraction**: Automatic extraction of text content from various file formats
- **OCR Capabilities**: Optical Character Recognition for image files
- **Content Analysis**: Parsed content is integrated into AI prompts for personalized suggestions
- **Multi-file Support**: Process multiple documents simultaneously

### Additional AI-Powered Features
- **Google Scholar Profile Analysis**: Scrapes and analyzes publications, citations, research areas, and trends to tailor topic suggestions
- **Topic Similarity Search**: Matches student interests with instructor topics using TF-IDF vectorization and cosine similarity
- **Research Paper Retrieval**: Fetches relevant papers from arXiv based on generated topics and keywords
- **Keyword Extraction**: Uses AI and traditional NLP to extract domain-specific terms for improved recommendations
- **Context Management**: Separates and prioritizes relevant user data for research vs. course project generation to improve accuracy
- **Error Handling & Fallbacks**: Automatic retries, fallback methods, and graceful degradation when APIs or parsing fail

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- OpenAI API key
- Tesseract OCR (for image processing)

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd IntelliTopic
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r intellitopic/requirements.txt
   ```

4. **Install Tesseract OCR**:
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`

5. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

6. **Run the application**:
   ```bash
   cd intellitopic
   uvicorn main:app --reload
   ```

7. **Access the application**:
   Open your browser and navigate to `http://localhost:8000`

## 📖 Usage

### For Students
1. Select "Student" as your role
2. Choose "Research" mode for thesis topics or "Non-Research" for course ideas
3. Describe your academic background, interests, and goals
4. **Optional**: Upload relevant documents (CV, research papers, course materials)
5. Click "Generate Ideas" to receive personalized suggestions

### For Professors
1. Select "Professor" as your role
2. Choose "Research" mode for research topics or "Non-Research" for course development
3. Describe your teaching background, research interests, and objectives
4. **Optional**: Upload relevant documents (syllabi, research papers, course materials)
5. Click "Generate Ideas" to receive personalized suggestions

### Document Upload Tips
- **CV/Resume**: Upload to get suggestions based on your experience and skills
- **Research Papers**: Include to generate ideas in your specific research area
- **Course Materials**: Upload existing syllabi or course outlines for related course ideas
- **Images**: Screenshots of diagrams, handwritten notes, or visual content
- **Data Files**: Excel files with research data or course information

## 🔧 Technical Architecture

### Backend (FastAPI)
- **Framework**: FastAPI for high-performance API development
- **AI Integration**: OpenAI GPT-3.5-turbo for content generation
- **File Processing**: Multi-format document parsing and text extraction
- **External Data Sources**: Google Scholar scraping, arXiv API integration
- **Context Management**: Mode-based context separation for accuracy
- **Template Engine**: Jinja2 for dynamic HTML generation

### Frontend
- **Modern UI**: Responsive design with gradient backgrounds and animations
- **File Upload**: Drag-and-drop interface with visual feedback
- **Interactive Elements**: Hover effects, loading states, and smooth transitions
- **Mobile Responsive**: Optimized for all device sizes

### Document Processing Pipeline
1. **File Upload**: Multiple file support with format validation
2. **Content Extraction**: Format-specific parsing (PDF, DOCX, Excel, Images, Text)
3. **Text Processing**: OCR for images, structured data extraction
4. **Context Integration**: Combined user input and document content
5. **AI Generation**: Enhanced prompts with document context
6. **Result Display**: Formatted output with document analysis summary

## 📁 Project Structure

```
IntelliTopic/
├── intellitopic/
│   ├── main.py              # FastAPI application and document processing
│   ├── requirements.txt     # Python dependencies
│   └── templates/
│       └── index.html       # Main application interface
├── .env                     # Environment variables (not in git)
├── .gitignore              # Git ignore rules
└── README.md               # Project documentation
```

## 🔒 Security & Privacy

- **Environment Variables**: Sensitive data stored in `.env` file (not committed to git)
- **File Processing**: Documents are processed in memory and not stored permanently
- **API Security**: OpenAI API key is securely managed through environment variables
- **Data Privacy**: No user data is stored or logged

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
