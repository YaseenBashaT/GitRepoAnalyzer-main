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
import git
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict, Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

def analyze_repository_metrics(repo_path):
    """Comprehensive repository analysis including git history, contributors, etc."""
    try:
        # First check if it's a valid git repository
        if not os.path.exists(os.path.join(repo_path, '.git')):
            st.warning("Repository doesn't appear to be a git repository. Analyzing file system only.")
            # Fall back to file system analysis only
            file_stats = analyze_file_system(repo_path)
            file_stats.update({
                'total_commits': 0,
                'author_stats': {},
                'daily_commits': {},
                'file_changes': {},
                'top_contributors': {},
                'commit_data': [],
                'repo_age_days': 0,
                'total_branches': 0,
                'total_tags': 0
            })
            return file_stats
        
        repo = git.Repo(repo_path)
        metrics = {}
        
        # Basic repository info
        metrics['repo_path'] = repo_path
        try:
            metrics['remote_url'] = repo.remotes.origin.url if repo.remotes else "Unknown"
        except:
            metrics['remote_url'] = "Unknown"
            
        try:
            metrics['current_branch'] = repo.active_branch.name
        except:
            metrics['current_branch'] = "Unknown"
            
        try:
            metrics['total_branches'] = len(list(repo.branches))
        except:
            metrics['total_branches'] = 0
            
        try:
            metrics['total_tags'] = len(list(repo.tags))
        except:
            metrics['total_tags'] = 0
        
        # Get all commits with error handling
        try:
            commits = list(repo.iter_commits('--all', max_count=1000))  # Limit to prevent memory issues
            metrics['total_commits'] = len(commits)
        except Exception as e:
            st.warning(f"Could not access commit history: {e}")
            commits = []
            metrics['total_commits'] = 0
        
        # Analyze commit history
        commit_data = []
        author_stats = defaultdict(int)
        daily_commits = defaultdict(int)
        file_changes = defaultdict(int)
        
        for commit in commits:
            try:
                commit_date = datetime.fromtimestamp(commit.committed_date)
                day_key = commit_date.strftime('%Y-%m-%d')
                
                commit_data.append({
                    'hash': commit.hexsha[:8],
                    'author': commit.author.name,
                    'email': commit.author.email,
                    'date': commit_date,
                    'message': commit.message.strip(),
                    'files_changed': len(commit.stats.files) if hasattr(commit, 'stats') else 0
                })
                
                author_stats[commit.author.name] += 1
                daily_commits[day_key] += 1
                
                # Count file changes with error handling
                try:
                    if hasattr(commit, 'stats'):
                        for file_path in commit.stats.files:
                            file_changes[file_path] += 1
                except:
                    pass  # Skip file changes if stats are not available
                    
            except Exception as e:
                # Skip problematic commits
                continue
        
        metrics['commit_data'] = commit_data
        metrics['author_stats'] = dict(author_stats)
        metrics['daily_commits'] = dict(daily_commits)
        metrics['file_changes'] = dict(file_changes)
        metrics['top_contributors'] = dict(sorted(author_stats.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Repository age
        if commits:
            try:
                first_commit = commits[-1]
                last_commit = commits[0]
                first_date = datetime.fromtimestamp(first_commit.committed_date)
                last_date = datetime.fromtimestamp(last_commit.committed_date)
                metrics['repo_age_days'] = (last_date - first_date).days
                metrics['first_commit_date'] = first_date
                metrics['last_commit_date'] = last_date
            except:
                metrics['repo_age_days'] = 0
        else:
            metrics['repo_age_days'] = 0
        
        # File system analysis
        file_stats = analyze_file_system(repo_path)
        metrics.update(file_stats)
        
        return metrics
        
    except git.exc.InvalidGitRepositoryError:
        st.warning("Invalid git repository. Analyzing file system only.")
        # Fall back to file system analysis
        file_stats = analyze_file_system(repo_path)
        file_stats.update({
            'total_commits': 0,
            'author_stats': {},
            'daily_commits': {},
            'file_changes': {},
            'top_contributors': {},
            'commit_data': [],
            'repo_age_days': 0,
            'total_branches': 0,
            'total_tags': 0
        })
        return file_stats
    except Exception as e:
        st.error(f"Error analyzing repository: {e}")
        st.info("Falling back to basic file system analysis...")
        
        # Fallback to basic file analysis
        try:
            file_stats = analyze_file_system(repo_path)
            file_stats.update({
                'total_commits': 0,
                'author_stats': {},
                'daily_commits': {},
                'file_changes': {},
                'top_contributors': {},
                'commit_data': [],
                'repo_age_days': 0,
                'total_branches': 0,
                'total_tags': 0
            })
            return file_stats
        except Exception as fallback_error:
            st.error(f"Complete analysis failed: {fallback_error}")
            return None

def analyze_file_system(repo_path):
    """Analyze file system structure and statistics"""
    file_stats = {
        'total_files': 0,
        'total_lines': 0,
        'file_types': defaultdict(int),
        'file_sizes': [],
        'language_stats': defaultdict(int),
        'largest_files': [],
        'directory_structure': defaultdict(int)
    }
    
    # Language extensions mapping
    language_map = {
        '.py': 'Python',
        '.js': 'JavaScript',
        '.jsx': 'JavaScript',
        '.ts': 'TypeScript',
        '.tsx': 'TypeScript',
        '.java': 'Java',
        '.c': 'C',
        '.cpp': 'C++',
        '.cc': 'C++',
        '.cxx': 'C++',
        '.cs': 'C#',
        '.php': 'PHP',
        '.rb': 'Ruby',
        '.go': 'Go',
        '.rs': 'Rust',
        '.swift': 'Swift',
        '.kt': 'Kotlin',
        '.scala': 'Scala',
        '.r': 'R',
        '.m': 'Objective-C',
        '.mm': 'Objective-C++',
        '.sh': 'Shell',
        '.bash': 'Bash',
        '.sql': 'SQL',
        '.html': 'HTML',
        '.htm': 'HTML',
        '.css': 'CSS',
        '.scss': 'SCSS',
        '.sass': 'Sass',
        '.less': 'Less',
        '.xml': 'XML',
        '.json': 'JSON',
        '.yaml': 'YAML',
        '.yml': 'YAML',
        '.md': 'Markdown',
        '.txt': 'Text',
        '.dockerfile': 'Docker',
        '.vue': 'Vue',
        '.svelte': 'Svelte',
        '.dart': 'Dart',
        '.lua': 'Lua',
        '.perl': 'Perl',
        '.pl': 'Perl'
    }
    
    for root, dirs, files in os.walk(repo_path):
        # Skip .git directory
        if '.git' in root:
            continue
            
        # Count directory depth
        depth = root.replace(repo_path, '').count(os.sep)
        file_stats['directory_structure'][depth] += len(files)
        
        for file in files:
            file_path = os.path.join(root, file)
            
            try:
                # Get file size
                file_size = os.path.getsize(file_path)
                file_stats['file_sizes'].append(file_size)
                
                # Get file extension
                _, ext = os.path.splitext(file)
                ext = ext.lower()
                file_stats['file_types'][ext] += 1
                
                # Map to language
                if ext in language_map:
                    file_stats['language_stats'][language_map[ext]] += 1
                
                # Count lines for text files
                if ext in ['.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.c', '.cpp', '.cs', 
                          '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala', '.r', '.m',
                          '.sh', '.bash', '.sql', '.html', '.htm', '.css', '.scss', '.xml',
                          '.json', '.yaml', '.yml', '.md', '.txt']:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            lines = len(f.readlines())
                            file_stats['total_lines'] += lines
                            
                            # Track largest files
                            file_stats['largest_files'].append({
                                'path': os.path.relpath(file_path, repo_path),
                                'lines': lines,
                                'size': file_size
                            })
                    except:
                        pass
                
                file_stats['total_files'] += 1
                
            except:
                continue
    
    # Sort largest files
    file_stats['largest_files'] = sorted(
        file_stats['largest_files'], 
        key=lambda x: x['lines'], 
        reverse=True
    )[:20]
    
    # Convert defaultdicts to regular dicts
    file_stats['file_types'] = dict(file_stats['file_types'])
    file_stats['language_stats'] = dict(file_stats['language_stats'])
    file_stats['directory_structure'] = dict(file_stats['directory_structure'])
    
    return file_stats

def display_repository_metrics(metrics, repo_name="default"):
    """Display comprehensive repository metrics dashboard"""
    st.header("📊 Repository Analytics Dashboard")
    
    # Basic Info Section
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📁 Total Files", f"{metrics['total_files']:,}")
    with col2:
        st.metric("📝 Total Lines", f"{metrics['total_lines']:,}")
    with col3:
        st.metric("🔀 Total Commits", f"{metrics['total_commits']:,}")
    with col4:
        st.metric("👥 Contributors", len(metrics['author_stats']))
    
    # Repository Timeline
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🌱 Repository Age", f"{metrics.get('repo_age_days', 0)} days")
    with col2:
        st.metric("🌿 Branches", metrics.get('total_branches', 0))
    with col3:
        st.metric("🏷️ Tags", metrics.get('total_tags', 0))
    
    # Commit Activity Chart
    st.subheader("📈 Commit Activity Over Time")
    if metrics.get('daily_commits') and len(metrics['daily_commits']) > 0:
        dates = list(metrics['daily_commits'].keys())
        commits = list(metrics['daily_commits'].values())
        
        df_commits = pd.DataFrame({
            'Date': pd.to_datetime(dates),
            'Commits': commits
        }).sort_values('Date')
        
        fig = px.line(df_commits, x='Date', y='Commits', 
                      title='Daily Commit Activity',
                      labels={'Commits': 'Number of Commits'})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key=f"commit_timeline_{repo_name}")
    else:
        st.info("📊 No commit data available for timeline analysis")
    
    # Contributors Analysis
    st.subheader("👥 Top Contributors")
    if metrics.get('top_contributors') and len(metrics['top_contributors']) > 0:
        contributors_df = pd.DataFrame(
            list(metrics['top_contributors'].items()), 
            columns=['Contributor', 'Commits']
        )
        
        fig = px.bar(contributors_df.head(10), x='Commits', y='Contributor', 
                     orientation='h', title='Top 10 Contributors by Commits')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True, key=f"contributors_chart_{repo_name}")
    else:
        st.info("👥 No contributor data available")
    
    # Language Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🗣️ Programming Languages")
        if metrics.get('language_stats') and len(metrics['language_stats']) > 0:
            lang_df = pd.DataFrame(
                list(metrics['language_stats'].items()), 
                columns=['Language', 'Files']
            )
            
            fig = px.pie(lang_df, values='Files', names='Language', 
                         title='Language Distribution by Files')
            st.plotly_chart(fig, use_container_width=True, key=f"language_pie_{repo_name}")
        else:
            st.info("🗣️ No programming language data available")
    
    with col2:
        st.subheader("📄 File Types")
        if metrics.get('file_types') and len(metrics['file_types']) > 0:
            # Filter out empty extensions and show top 10
            file_types_filtered = {k: v for k, v in metrics['file_types'].items() if k and v > 0}
            sorted_types = sorted(file_types_filtered.items(), key=lambda x: x[1], reverse=True)[:10]
            
            if sorted_types:
                types_df = pd.DataFrame(sorted_types, columns=['Extension', 'Count'])
                
                fig = px.bar(types_df, x='Extension', y='Count', 
                             title='Top File Extensions')
                st.plotly_chart(fig, use_container_width=True, key=f"file_types_chart_{repo_name}")
            else:
                st.info("📄 No file type data available")
        else:
            st.info("📄 No file type data available")
    
    # File Size Analysis
    st.subheader("📊 File Size Distribution")
    if metrics.get('file_sizes') and len(metrics['file_sizes']) > 0:
        sizes_mb = [size / (1024 * 1024) for size in metrics['file_sizes']]
        
        fig = px.histogram(x=sizes_mb, nbins=50, 
                          title='File Size Distribution (MB)',
                          labels={'x': 'File Size (MB)', 'y': 'Count'})
        st.plotly_chart(fig, use_container_width=True, key=f"file_size_hist_{repo_name}")
    else:
        st.info("📊 No file size data available")
    
    # Largest Files
    st.subheader("📄 Largest Files")
    if metrics['largest_files']:
        large_files_df = pd.DataFrame(metrics['largest_files'][:10])
        large_files_df['Size (KB)'] = large_files_df['size'] / 1024
        
        st.dataframe(
            large_files_df[['path', 'lines', 'Size (KB)']].round(2),
            use_container_width=True
        )
    
    # Recent Commits
    st.subheader("🕒 Recent Commits")
    if metrics['commit_data']:
        recent_commits = metrics['commit_data'][:20]
        commits_df = pd.DataFrame(recent_commits)
        commits_df['date'] = pd.to_datetime(commits_df['date'])
        
        # Format for display
        display_df = commits_df[['hash', 'author', 'date', 'message', 'files_changed']].copy()
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['message'] = display_df['message'].str[:100] + '...'
        
        st.dataframe(display_df, use_container_width=True)
    
    # Repository Health Metrics
    st.subheader("🏥 Repository Health")
    health_col1, health_col2, health_col3 = st.columns(3)
    
    with health_col1:
        # Calculate commit frequency
        if metrics.get('repo_age_days', 0) > 0:
            commit_frequency = metrics['total_commits'] / max(metrics['repo_age_days'], 1)
            st.metric("📊 Commits/Day", f"{commit_frequency:.2f}")
        
    with health_col2:
        # Calculate average file size
        if metrics['file_sizes']:
            avg_size = sum(metrics['file_sizes']) / len(metrics['file_sizes']) / 1024  # KB
            st.metric("📐 Avg File Size", f"{avg_size:.1f} KB")
    
    with health_col3:
        # Calculate lines per file
        if metrics['total_files'] > 0:
            lines_per_file = metrics['total_lines'] / metrics['total_files']
            st.metric("📄 Lines/File", f"{lines_per_file:.1f}")

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
    
    # Add sub-navigation tabs after repo URL input
    tab1, tab2 = st.tabs(["📊 Repository Metrics", "💬 AI Chat Analysis"])
    
    repo_name = repo_url.split("/")[-1]
    
    # Clear old cache files periodically
    clear_old_cache()
    
    with tab1:
        st.header("📊 Repository Analytics Dashboard")
        
        if st.button("🔍 Analyze Repository Metrics", type="primary"):
            with st.spinner("🔄 Cloning and analyzing repository..."):
                with tempfile.TemporaryDirectory() as local_path:
                    if clone_git_repo(repo_url, local_path):
                        metrics = analyze_repository_metrics(local_path)
                        if metrics:
                            # Store metrics in session state for persistence
                            st.session_state[f'metrics_{repo_name}'] = metrics
                            st.success("✅ Repository analysis completed!")
                        else:
                            st.error("Failed to analyze repository metrics.")
                    else:
                        st.error("Failed to clone repository. Please check the URL.")
        
        # Display cached metrics if available
        if f'metrics_{repo_name}' in st.session_state:
            display_repository_metrics(st.session_state[f'metrics_{repo_name}'], repo_name)
        else:
            st.info("👆 Click the button above to analyze repository metrics")
    
    with tab2:
        st.header("💬 AI-Powered Repository Analysis")
        
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
        if 'question_context' not in locals() or question_context is None:
            st.error("Please load repository data first.")
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