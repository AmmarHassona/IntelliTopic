# IntelliTopic - AI-Powered Academic Idea Generator

IntelliTopic is a sophisticated web application that generates personalized academic ideas for students and professors using AI. The application supports multi-modality input, allowing users to upload documents that are parsed and analyzed to create more relevant and personalized suggestions.

## 🌟 Features

### Core Functionality
- **Dual User Types**: Support for both students and professors
- **Two Generation Modes**:
  - **Research Mode**: Generate comprehensive thesis topics
  - **Non-Research Mode**: Generate creative course/module ideas
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
- **Text Extraction**: Automatic extraction of text content from various file formats
- **OCR Capabilities**: Optical Character Recognition for image files
- **Content Analysis**: Parsed content is integrated into AI prompts for personalized suggestions
- **Multi-file Support**: Process multiple documents simultaneously

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
- **File Processing**: Multi-format document parsing and text extraction
- **AI Integration**: OpenAI API for intelligent content generation
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

## 🛠️ Customization

### Adding New File Formats
To support additional file formats, add new parsing functions in `main.py`:

```python
async def parse_new_format(file_content: bytes) -> str:
    # Add your parsing logic here
    return extracted_text
```

### Modifying AI Prompts
Customize the generation prompts in the `generate` function to match your specific needs.

### UI Customization
Modify the CSS styles in `templates/index.html` to change the appearance and branding.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for providing the GPT API
- FastAPI for the excellent web framework
- The open-source community for the various parsing libraries used

## 📞 Support

For questions, issues, or feature requests, please open an issue on the GitHub repository.