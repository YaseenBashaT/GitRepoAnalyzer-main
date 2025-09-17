import os
import tempfile
from typing import List, Any, Optional, Mapping, Dict
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import Generation, LLMResult
from pydantic import Field, PrivateAttr
from repo_reader import clone_git_repo, load_and_index_files
from questions import QuestionContext, ask_question
from utility import format_questions
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
import time
import random
import hashlib
import pickle
import shutil
import re

load_dotenv()

# Initialize session state
if 'conversation_count' not in st.session_state:
    st.session_state.conversation_count = 0
if 'current_repo' not in st.session_state:
    st.session_state.current_repo = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = ""
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'cached_repos' not in st.session_state:
    st.session_state.cached_repos = {}
if 'current_question_context' not in st.session_state:
    st.session_state.current_question_context = None

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
model_name = "llama-3.3-70b-versatile"  # Updated to current supported model

# Create cache directory
CACHE_DIR = os.path.join(os.getcwd(), "repo_cache")
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_repo_hash(repo_url):
    """Generate a hash for the repository URL"""
    return hashlib.md5(repo_url.encode()).hexdigest()

def get_cache_path(repo_url):
    """Get the cache directory path for a repository"""
    repo_hash = get_repo_hash(repo_url)
    return os.path.join(CACHE_DIR, repo_hash)

def is_repo_cached(repo_url):
    """Check if repository is already cached"""
    cache_path = get_cache_path(repo_url)
    return os.path.exists(cache_path) and os.path.exists(os.path.join(cache_path, "cache_data.pkl"))

def save_repo_cache(repo_url, index, document, file_type_count, file_names):
    """Save repository processing results to cache"""
    cache_path = get_cache_path(repo_url)
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    
    cache_data = {
        'index': index,
        'document': document,
        'file_type_count': file_type_count,
        'file_names': file_names,
        'timestamp': time.time()
    }
    
    with open(os.path.join(cache_path, "cache_data.pkl"), 'wb') as f:
        pickle.dump(cache_data, f)

def load_repo_cache(repo_url):
    """Load repository processing results from cache"""
    cache_path = get_cache_path(repo_url)
    cache_file = os.path.join(cache_path, "cache_data.pkl")
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        return cache_data['index'], cache_data['document'], cache_data['file_type_count'], cache_data['file_names']
    return None, None, None, None

def clear_old_cache(max_age_hours=24):
    """Clear cache files older than specified hours"""
    if not os.path.exists(CACHE_DIR):
        return
    
    current_time = time.time()
    for cache_folder in os.listdir(CACHE_DIR):
        cache_path = os.path.join(CACHE_DIR, cache_folder)
        cache_file = os.path.join(cache_path, "cache_data.pkl")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                if current_time - cache_data['timestamp'] > max_age_hours * 3600:
                    shutil.rmtree(cache_path)
                    print(f"Cleared old cache for {cache_folder}")
            except:
                # If we can't read the cache file, remove it
                shutil.rmtree(cache_path)

# Copy functionality removed - cleaner interface without copy buttons

def parse_and_display_response(response_text):
    """Parse response and display with ChatGPT-like copy functionality"""
    # Split response into parts (text and code blocks)
    parts = re.split(r'```(\w*)\n?(.*?)```', response_text, flags=re.DOTALL)
    code_blocks = []
    
    # First pass: collect all code blocks
    for i, part in enumerate(parts):
        if i % 3 == 2:  # Code block content
            language = parts[i-1] if parts[i-1] else "text"
            code_blocks.append((language, part.strip()))
    
    # Second pass: display content
    for i, part in enumerate(parts):
        if i % 3 == 0:  # Regular text
            if part.strip():
                st.markdown(part.strip())
        elif i % 3 == 1:  # Language identifier
            continue
        else:  # Code block content
            language = parts[i-1] if parts[i-1] else "text"
            code_content = part.strip()
            
            # Display code block without copy button
            with st.container():
                # Header with language only (no copy button)
                st.markdown(f"**{language.upper() if language else 'CODE'}**")
                
                # Display the actual code with syntax highlighting
                st.code(code_content, language=language)
                
                # Add a subtle separator
                st.markdown("---")
    
    # Multiple code blocks display - simplified without copy functionality

def display_enhanced_answer(answer):
    """Display answer with formatting but no copy functionality"""
    # Add custom CSS for better styling (keeping visual improvements)
    st.markdown("""
    <style>
    .code-block-container {
        background-color: #f6f8fa;
        border: 1px solid #d0d7de;
        border-radius: 6px;
        margin: 10px 0;
        position: relative;
    }
    .code-header {
        background-color: #f6f8fa;
        border-bottom: 1px solid #d0d7de;
        padding: 8px 16px;
        font-size: 12px;
        color: #656d76;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        # Check if the answer contains code blocks
        if '```' in answer:
            parse_and_display_response(answer)
        else:
            # Regular text answer without copy option
            st.markdown(answer)

from pydantic import Field, PrivateAttr

class GroqLLM(BaseLLM):
    model_name: str = Field(default=model_name)
    _client: Groq = PrivateAttr()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = Groq(api_key=GROQ_API_KEY)
        
    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None) -> LLMResult:
        generations = []
        for prompt in prompts:
            # Retry logic for handling service unavailable errors
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    response = self._client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that analyzes code repositories."},
                            {"role": "user", "content": prompt}
                        ],
                        timeout=30  # Add timeout
                    )
                    generations.append([Generation(text=response.choices[0].message.content)])
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    error_message = str(e)
                    
                    # Handle specific error types
                    if "503" in error_message or "Service unavailable" in error_message:
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                            print(f"Groq service temporarily unavailable. Retrying in {wait_time:.1f} seconds... (Attempt {attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                            continue
                        else:
                            # Fallback response for final failure
                            fallback_response = f"I apologize, but the AI service is currently experiencing issues. However, I can see that your repository has been successfully cloned and indexed. You can try asking your question again in a moment, or here's what I can tell you from the repository structure: The repository appears to contain code files that have been successfully processed. Please try your question again shortly when the service is restored."
                            generations.append([Generation(text=fallback_response)])
                            st.error("AI service temporarily unavailable. Please try again in a moment.")
                            break
                    
                    elif "rate limit" in error_message.lower():
                        if attempt < max_retries - 1:
                            wait_time = 60  # Wait longer for rate limits
                            print(f"Rate limit reached. Waiting {wait_time} seconds before retry...")
                            time.sleep(wait_time)
                            continue
                        else:
                            fallback_response = "Rate limit exceeded. Please wait a moment before asking another question."
                            generations.append([Generation(text=fallback_response)])
                            st.warning("Rate limit reached. Please wait a moment before trying again.")
                            break
                    
                    else:
                        # Other errors - log to console, only show critical ones to user
                        print(f"Error occurred: {error_message}. Retrying... (Attempt {attempt + 1}/{max_retries})")
                        
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                        else:
                            # Only show user-friendly error for critical failures
                            if "authentication" in error_message.lower() or "api_key" in error_message.lower():
                                st.error("Authentication error. Please check your API key configuration.")
                                fallback_response = "Authentication error. Please check your API key and try again."
                            elif "timeout" in error_message.lower():
                                st.error("Request timed out. Please try again.")
                                fallback_response = "Request timed out. Please try again with a shorter question."
                            else:
                                st.error("Service temporarily unavailable. Please try again later.")
                                fallback_response = "I encountered a temporary issue. Please try rephrasing your question or try again in a moment."
                            
                            generations.append([Generation(text=fallback_response)])
                            break
        
        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "groq"
def process_repository_fresh(repo_url, repo_name):
    """Process repository fresh when not cached"""
    st.info("🔄 Cloning and processing repository for the first time...")
    
    with tempfile.TemporaryDirectory() as local_path:
        if clone_git_repo(repo_url, local_path):
            index, document, file_type_count, file_names = load_and_index_files(local_path)
            
            if index is None:
                st.error("No documents were found to index in this repository.")
                return None, None, None, None, None

            print("Repo cloned.....Indexing Files")
            llm = GroqLLM()

            template = '''
            You are an expert code analyst assistant. You have access to the repository content and our conversation history.

            Repository: {repo_name} ({repo_url})
            File Types: {file_type_count}
            Files: {file_names}

            CONVERSATION HISTORY:
            {conversation_history}

            REPOSITORY CONTENT:
            {numbered_documents}

            CURRENT QUESTION: {question}

            Instructions:
            1. Use the conversation history to understand context and references to previous discussions
            2. When the user asks follow-up questions (like "explain more", "what about X", "how does that work"), refer back to previous answers
            3. If the user references "it", "that", "this", or "the previous answer", use conversation history for context
            4. Provide detailed, contextual answers based on both the repository content and our ongoing conversation
            5. If you're unsure about something, say "I am not sure"
            6. For follow-up questions, build upon previous answers rather than starting fresh
            7. IMPORTANT: Format code snippets and commands in markdown code blocks with appropriate language tags
            8. Use ```bash for terminal commands, ```python for Python code, ```javascript for JS, etc.
            9. Only put actual runnable commands and code in code blocks, not explanatory text
            10. Make your responses copy-friendly for developers who need to run commands or use code

            Answer:
            '''

            prompt = PromptTemplate(
                template=template,
                input_variables=["repo_name","repo_url","conversation_history","numbered_documents","question","file_type_count","file_names"]
            )

            llm_chain = LLMChain(prompt=prompt, llm=llm)

            question_context = QuestionContext(
                index,
                document,
                llm_chain,
                model_name,
                repo_name,
                repo_url,
                st.session_state.conversation_history,
                file_type_count,
                file_names
            )
            
            # Save to cache
            save_repo_cache(repo_url, index, document, file_type_count, file_names)
            
            # Cache in session state for faster access
            st.session_state.cached_repos[repo_url] = {
                'index': index,
                'document': document,
                'file_type_count': file_type_count,
                'file_names': file_names,
                'question_context': question_context
            }
            
            st.success(f"✅ Repository '{repo_name}' processed and cached successfully!")
            return index, document, file_type_count, file_names, question_context
        else:
            st.error("Failed to clone repository. Please check the URL and try again.")
            return None, None, None, None, None

def main():
    st.title("🚀 Advanced Git Repository Analyzer")
    st.markdown("### AI-Powered Code Analysis with Real-Time Metrics Dashboard")
    
    # Remove custom CSS for copy buttons (keeping only basic styling)
    st.markdown("""
    <style>
    .stCode {
        position: relative;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add cache management in sidebar
    with st.sidebar:
        st.markdown("### 📁 Cache Management")
        if st.button("🗑️ Clear All Cache"):
            if os.path.exists(CACHE_DIR):
                shutil.rmtree(CACHE_DIR)
                os.makedirs(CACHE_DIR)
            st.session_state.cached_repos = {}
            st.success("Cache cleared!")
        
        # Show cache status
        cache_count = len(os.listdir(CACHE_DIR)) if os.path.exists(CACHE_DIR) else 0
        st.info(f"📊 Cached repositories: {cache_count}")
    
    repo_url = st.text_input(
        "Github URL",
        placeholder="Enter the Github Url of the Repo",
        key="repo_url_input"
    )
    
    # Reset conversation if repo changes
    if repo_url != st.session_state.current_repo:
        st.session_state.conversation_count = 0
        st.session_state.conversation_history = ""
        st.session_state.qa_history = []
        st.session_state.current_repo = repo_url
        st.session_state.current_question_context = None  # Reset cached context
    
    if not repo_url:  # Skip if no URL is provided
        return
    
    repo_name = repo_url.split("/")[-1]
    
    # Clear old cache files periodically
    clear_old_cache()
    
    # Check if repository data is already processed and cached
    if repo_url in st.session_state.cached_repos:
        # Use cached data
        cached_data = st.session_state.cached_repos[repo_url]
        index = cached_data['index']
        document = cached_data['document']
        file_type_count = cached_data['file_type_count']
        file_names = cached_data['file_names']
        question_context = cached_data['question_context']
        st.success(f"✅ Repository '{repo_name}' loaded from memory cache!")
        
    elif is_repo_cached(repo_url):
        # Load from disk cache
        st.info("📂 Loading repository from disk cache...")
        index, document, file_type_count, file_names = load_repo_cache(repo_url)
        
        if index is not None:
            # Create question context
            llm = GroqLLM()
            template = '''
            You are an expert code analyst assistant. You have access to the repository content and our conversation history.

            Repository: {repo_name} ({repo_url})
            File Types: {file_type_count}
            Files: {file_names}

            CONVERSATION HISTORY:
            {conversation_history}

            REPOSITORY CONTENT:
            {numbered_documents}

            CURRENT QUESTION: {question}

            Instructions:
            1. Use the conversation history to understand context and references to previous discussions
            2. When the user asks follow-up questions (like "explain more", "what about X", "how does that work"), refer back to previous answers
            3. If the user references "it", "that", "this", or "the previous answer", use conversation history for context
            4. Provide detailed, contextual answers based on both the repository content and our ongoing conversation
            5. If you're unsure about something, say "I am not sure"
            6. For follow-up questions, build upon previous answers rather than starting fresh
            7. IMPORTANT: Format code snippets and commands in markdown code blocks with appropriate language tags
            8. Use ```bash for terminal commands, ```python for Python code, ```javascript for JS, etc.
            9. Only put actual runnable commands and code in code blocks, not explanatory text
            10. Make your responses copy-friendly for developers who need to run commands or use code

            Answer:
            '''

            prompt = PromptTemplate(
                template=template,
                input_variables=["repo_name","repo_url","conversation_history","numbered_documents","question","file_type_count","file_names"]
            )

            llm_chain = LLMChain(prompt=prompt, llm=llm)

            question_context = QuestionContext(
                index,
                document,
                llm_chain,
                model_name,
                repo_name,
                repo_url,
                st.session_state.conversation_history,
                file_type_count,
                file_names
            )
            
            # Cache in session state for faster access
            st.session_state.cached_repos[repo_url] = {
                'index': index,
                'document': document,
                'file_type_count': file_type_count,
                'file_names': file_names,
                'question_context': question_context
            }
            
            st.success(f"✅ Repository '{repo_name}' loaded from disk cache!")
        else:
            st.error("Failed to load cached data. Will re-process repository...")
            # Fallback to normal processing
            index, document, file_type_count, file_names, question_context = process_repository_fresh(repo_url, repo_name)
    else:
        # Process repository fresh
        index, document, file_type_count, file_names, question_context = process_repository_fresh(repo_url, repo_name)
    
    # If processing failed, return early
    if question_context is None:
        return

    # Create a unique key for this question input
    question_key = f"question_input_{st.session_state.conversation_count}_{repo_name}"
    
    # Initialize the session state for this question if it doesn't exist
    if question_key not in st.session_state:
        st.session_state[question_key] = ""
    
    # Store previous value to check for changes
    previous_value = st.session_state[question_key]
    
    user_question = st.text_input(
        "Ask a question about the repository: (Press Enter or click Submit)",
        key=question_key,
        value=previous_value  # Use the stored value
    )
    
    btn = st.button(
        "Submit",
        key=f"submit_button_{st.session_state.conversation_count}_{repo_name}"
    )

    # Check if either Enter was pressed (text changed) or Submit was clicked
    question_changed = user_question != previous_value

    if user_question.lower() == "exit()":
        st.warning("Session ended")
        return

    # Display conversation history (without copy buttons)
    if hasattr(st.session_state, 'qa_history') and st.session_state.qa_history:
        st.subheader("💬 Conversation History")
        for i, (q, a) in enumerate(st.session_state.qa_history):
            with st.expander(f"Q{i+1}: {q[:100]}{'...' if len(q) > 100 else ''}", expanded=(i == len(st.session_state.qa_history) - 1)):
                st.write(f"**Question:** {q}")
                st.write(f"**Answer:**")
                
                # Enhanced answer display without copy functionality
                display_enhanced_answer(a)
                
        st.divider()

    if btn or (question_changed and user_question):
        try:
            with st.spinner("Processing your question..."):
                user_question = format_questions(user_question)
                
                # Update the question context with current conversation history before asking
                question_context.conversation_history = st.session_state.conversation_history
                
                answer = ask_question(user_question, question_context)
                
                # Initialize qa_history if it doesn't exist
                if 'qa_history' not in st.session_state:
                    st.session_state.qa_history = []
                
                # Add to conversation history
                st.session_state.qa_history.append((user_question, answer))
                
                # Format conversation history for better context
                formatted_history = ""
                for i, (q, a) in enumerate(st.session_state.qa_history):
                    formatted_history += f"===== Previous Conversation {i+1} =====\n"
                    formatted_history += f"User: {q}\n"
                    formatted_history += f"Assistant: {a}\n\n"
                
                st.session_state.conversation_history = formatted_history
                st.session_state.conversation_count += 1
                
                # Display the new answer immediately with enhanced formatting (no copy buttons)
                st.success("Latest Answer:")
                display_enhanced_answer(answer)
                
                # Force a rerun to update the conversation history display
                st.rerun()
                
        except Exception as ex:
            error_message = str(ex)
            print(f"An error occurred: {ex}")  # Log full error to console
            
            # Show user-friendly error messages
            if "503" in error_message or "Service unavailable" in error_message:
                st.error("AI service is temporarily unavailable. Please try again in a moment.")
            elif "rate limit" in error_message.lower():
                st.error("Rate limit reached. Please wait before asking another question.")
            elif "authentication" in error_message.lower() or "api_key" in error_message.lower():
                st.error("Authentication issue. Please check your API configuration.")
            elif "timeout" in error_message.lower():
                st.error("Request timed out. Please try with a shorter question.")
            else:
                st.error("Something went wrong. Please try again or rephrase your question.")
            
            st.session_state.conversation_count += 1  # Increment counter even on error

if __name__ == "__main__":
    main()