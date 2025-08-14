# Git Repository Analyzer

An AI-powered tool that analyzes GitHub repositories and answers questions about their content using the Groq API.

## Features

- Clone and analyze any public GitHub repository
- Ask questions about the repository's code, structure, and purpose
- Get AI-powered responses using Groq's LLM
- Interactive Streamlit interface
- Maintains conversation history for context-aware responses

## Requirements

- Python 3.8+
- Groq API key
- Required Python packages (see requirements.txt)

## Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/GitRepoAnalyzer.git
cd GitRepoAnalyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file and add your Groq API key:
```
GROQ_API_KEY=your_api_key_here
```

4. Run the application:
```bash
streamlit run main.py
```

## Usage

1. Enter a GitHub repository URL
2. Ask questions about the repository
3. Get AI-powered answers about the code

## Project Structure

- `main.py` - Main application and Streamlit interface
- `repo_reader.py` - Repository cloning and file processing
- `questions.py` - Question handling and context management
- `utility.py` - Utility functions for text processing

## License

MIT License
