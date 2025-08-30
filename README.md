# AI-Chatbot
AI chatbot using Langchain to connect large language models (LLMs) with external data sources and tools, enabling advanced functionalities like memory and retrieval-augmented generation (RAG).

## Multi-Format Document Support

The AI-Chatbot now supports text retrieval from multiple document formats:

- **PDF documents** (.pdf) - Portable Document Format files
- **Text files** (.txt) - Plain text documents  
- **Word documents** (.docx) - Microsoft Word documents
- **Excel spreadsheets** (.xlsx, .xls) - Microsoft Excel files
- **CSV files** (.csv) - Comma-separated values data files

## Features

- **Multi-format document processing**: Automatically detects and processes various document types
- **Semantic search**: Advanced text retrieval capabilities across all supported formats
- **Context-aware responses**: Understands and responds to questions about document content
- **Memory management**: Maintains conversation history for better context
- **External tool integration**: Includes calculator and web search capabilities

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Add your documents to the appropriate directories:
   - `./pdf/` - for PDF files
   - `./data/samples/` - for other document formats (txt, docx, xlsx, csv)

3. Configure your LLM endpoint in `app.py` (currently set for LM Studio)

4. Run the application:
   ```bash
   python app.py
   ```

The chatbot will automatically load all supported document types and make them available for question answering through the Document Knowledge Base tool.
