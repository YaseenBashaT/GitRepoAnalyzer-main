import os
import tempfile
from typing import List, Any, Optional, Mapping, Dict
from langchain_core.prompts import PromptTemplate
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
import networkx as nx
import ast

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

def apply_modern_styling():
    """Apply clean, minimalistic CSS styling to the application"""
    st.markdown("""
    <style>
    /* Import clean font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    /* Global Styles with professional background */
    .main {
        font-family: 'Inter', sans-serif;
        padding: 1rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    
    /* Professional page background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Professional sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    /* Professional metric containers */
    .stMetric {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Professional data frames */
    .stDataFrame {
        border: 1px solid #dee2e6;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Simple Header with professional colors */
    .modern-header {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        margin: 0 0 2rem 0;
        text-align: center;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .header-title {
        color: #2c3e50;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    .header-subtitle {
        color: #5d6d7e;
        font-size: 1rem;
        font-weight: 400;
        margin: 0.5rem 0 0 0;
    }
    
    /* Professional Input Section */
    .input-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Enhanced Professional Metrics */
    .enhanced-metric {
        background: linear-gradient(135deg, #ffffff 0%, #f1f3f4 100%);
        border: 1px solid #d1d5db;
        border-radius: 6px;
        padding: 1rem;
        text-align: center;
        display: inline-block;
        min-width: 120px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    
    .enhanced-metric:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .metric-icon {
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
        color: #3498db;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 0.25rem 0;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #5d6d7e;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Professional Architecture Section */
    .architecture-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .architecture-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e9ecef;
    }
    
    .architecture-icon {
        font-size: 1.2rem;
        color: #3498db;
    }
    
    .architecture-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #2c3e50 !important;
        margin: 0;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Professional Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(52, 152, 219, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2980b9 0%, #3498db 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(52, 152, 219, 0.4);
    }
    
    /* Professional Cards with classic color palette */
    .insight-card {
        background: linear-gradient(135deg, #ecf0f1 0%, #bdc3c7 100%);
        border-left: 4px solid #3498db;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border-radius: 6px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
        color: #2c3e50;
    }
    
    .insight-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .insight-warning {
        border-left-color: #e74c3c;
        background: linear-gradient(135deg, #fadbd8 0%, #f1948a 100%);
        color: #722f37;
    }
    
    .insight-success {
        border-left-color: #27ae60;
        background: linear-gradient(135deg, #d5f4e6 0%, #82e0aa 100%);
        color: #1e8449;
    }
    
    .insight-info {
        border-left-color: #3498db;
        background: linear-gradient(135deg, #d6eaf8 0%, #85c1e9 100%);
        color: #1f618d;
    }
    
    /* Professional input styling */
    .stTextInput > div > div > input {
        border: 2px solid #bdc3c7;
        border-radius: 6px;
        padding: 0.75rem;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3498db;
        box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
    }
    
    .stSelectbox > div > div > div {
        border: 2px solid #bdc3c7;
        border-radius: 6px;
        transition: border-color 0.3s ease;
    }
    
    .stSelectbox > div > div > div:focus-within {
        border-color: #3498db;
        box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
    }
    
    /* Typography - Professional color palette */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    
    /* Professional heading hierarchy */
    .stApp h1, .stMarkdown h1 {
        color: #2c3e50;  /* Deep navy blue */
        font-weight: 700;
        font-family: 'Inter', sans-serif;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    .stApp h2, .stMarkdown h2 {
        color: #34495e;  /* Slate gray */
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        margin-bottom: 1rem;
    }
    
    .stApp h3, .stMarkdown h3 {
        color: #5d6d7e;  /* Cool gray */
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        margin-bottom: 0.8rem;
    }
    
    .stApp h4, .stMarkdown h4, .stApp h5, .stMarkdown h5, .stApp h6, .stMarkdown h6 {
        color: #34495e;  /* Darker gray instead of muted */
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        margin-bottom: 0.6rem;
    }
    
    /* Headings on dark/colored backgrounds */
    .insight-card h1, .insight-card h2, .insight-card h3, .insight-card h4, .insight-card h5, .insight-card h6,
    .architecture-section h1, .architecture-section h2, .architecture-section h3, .architecture-section h4, .architecture-section h5, .architecture-section h6 {
        color: #2c3e50 !important;  /* Force dark color for architecture sections */
    }
    
    /* Headings in gradient backgrounds */
    [style*="background: linear-gradient"] h1,
    [style*="background: linear-gradient"] h2,
    [style*="background: linear-gradient"] h3,
    [style*="background: linear-gradient"] h4,
    [style*="background: linear-gradient"] h5,
    [style*="background: linear-gradient"] h6 {
        color: #ffffff;
    }
    
    /* Alert headings inherit appropriate colors */
    .stAlert h1, .stAlert h2, .stAlert h3, .stAlert h4, .stAlert h5, .stAlert h6 {
        color: inherit;
    }
    
    p, div, span {
        font-family: 'Inter', sans-serif;
        color: #2c3e50;
    }
    
    /* Ensure proper text contrast everywhere */
    .stApp, .stApp p, .stApp div, .stApp span {
        color: #2c3e50;
    }
    
    /* Fix small text visibility */
    small, .small {
        color: #34495e !important;
        font-weight: 500;
    }
    
    /* Improve metric text contrast */
    .metric-value {
        color: #2c3e50 !important;
    }
    
    /* Fix chart annotations */
    .js-plotly-plot .annotation-text {
        color: #2c3e50 !important;
    }
    
    /* Ensure all text is visible */
    .stDataFrame table {
        color: #2c3e50 !important;
    }
    
    .stDataFrame th {
        background-color: #34495e !important;
        color: #ffffff !important;
        font-weight: 600;
    }
    
    .stDataFrame td {
        color: #2c3e50 !important;
    }
    
    /* Fix expander text */
    .streamlit-expanderHeader {
        color: #2c3e50 !important;
        font-weight: 600;
    }
    
    /* Fix alert text contrast */
    .stAlert {
        border-radius: 6px;
    }
    
    .stSuccess {
        background-color: #d4edda !important;
        color: #155724 !important;
        border-color: #c3e6cb !important;
    }
    
    .stWarning {
        background-color: #fff3cd !important;
        color: #856404 !important;
        border-color: #ffeaa7 !important;
    }
    
    .stInfo {
        background-color: #d1ecf1 !important;
        color: #0c5460 !important;
        border-color: #bee5eb !important;
    }
    
    .stError {
        background-color: #f8d7da !important;
        color: #721c24 !important;
        border-color: #f5c6cb !important;
    }
    </style>
    """, unsafe_allow_html=True)

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
            st.warning("⚠️ **Not a Git Repository**: Repository doesn't appear to be a git repository. Analyzing file system only.")
            st.info("💡 **Note**: Limited analysis available - commit history, contributors, and git-specific metrics will not be available.")
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
            st.warning(f"⚠️ **Repository Access Issue**: Could not access commit history - {str(e)}")
            st.info("💡 **Tip**: This often happens with shallow clones, corrupted repositories, or access permission issues.")
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
        st.error(f"❌ **Repository Analysis Failed**: {str(e)}")
        with st.expander("🔧 Troubleshooting Tips"):
            st.markdown("""
            **Common causes and solutions:**
            - **Access denied**: Check repository URL and permissions
            - **Network issues**: Verify internet connection and firewall settings
            - **Large repositories**: Try with a smaller repository first
            - **Corrupted repository**: Re-clone the repository
            - **Authentication required**: Ensure you have access to private repositories
            """)
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
            st.error(f"💥 **Complete Analysis Failed**: {str(fallback_error)}")
            with st.expander("🆘 Recovery Options"):
                st.markdown("""
                **Try these alternatives:**
                - **Check path**: Ensure the repository path is correct and accessible
                - **File permissions**: Verify you have read access to all files
                - **Try a different repository**: Test with a simpler project structure
                - **Contact support**: If the issue persists, report this error
                """)
            return None

def generate_architecture_diagram(repo_path):
    """Generate interactive architecture diagram showing module dependencies"""
    try:
        # Build dependency graph
        G = nx.DiGraph()
        module_info = {}
        
        # Language-specific import patterns
        import_patterns = {
            '.py': [
                r'^from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import',
                r'^import\s+([a-zA-Z_][a-zA-Z0-9_.]*)',
            ],
            '.js': [
                r'from\s+["\']([^"\']+)["\']',
                r'import\s+.*\s+from\s+["\']([^"\']+)["\']',
                r'require\s*\(\s*["\']([^"\']+)["\']\s*\)',
            ],
            '.jsx': [
                r'from\s+["\']([^"\']+)["\']',
                r'import\s+.*\s+from\s+["\']([^"\']+)["\']',
            ],
            '.ts': [
                r'from\s+["\']([^"\']+)["\']',
                r'import\s+.*\s+from\s+["\']([^"\']+)["\']',
            ],
            '.tsx': [
                r'from\s+["\']([^"\']+)["\']',
                r'import\s+.*\s+from\s+["\']([^"\']+)["\']',
            ],
            '.java': [
                r'import\s+([a-zA-Z_][a-zA-Z0-9_.]*);',
            ],
            '.cpp': [
                r'#include\s+[<"]([^>"]+)[>"]',
            ],
            '.c': [
                r'#include\s+[<"]([^>"]+)[>"]',
            ]
        }
        
        # Analyze files and build dependency graph
        for root, dirs, files in os.walk(repo_path):
            # Skip .git and other hidden directories
            if any(skip in root for skip in ['.git', '__pycache__', 'node_modules', '.venv', 'venv']):
                continue
                
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                if file_ext in import_patterns:
                    try:
                        # Create module name from file path
                        rel_path = os.path.relpath(file_path, repo_path)
                        module_name = rel_path.replace(os.sep, '.').replace('/', '.')
                        if file_ext in ['.py']:
                            module_name = module_name[:-3]  # Remove .py extension
                        elif file_ext in ['.js', '.jsx', '.ts', '.tsx']:
                            module_name = module_name[:-len(file_ext)]
                        
                        # Store simple name for easier matching
                        simple_name = os.path.splitext(os.path.basename(file))[0]
                        
                        # Read file content
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # Count lines and complexity estimate
                        lines_count = len(content.splitlines())
                        
                        # Simple complexity estimate based on keywords
                        complexity_keywords = ['if', 'for', 'while', 'try', 'catch', 'switch', 'case']
                        complexity_score = sum(content.lower().count(keyword) for keyword in complexity_keywords)
                        
                        # Store module information
                        module_info[module_name] = {
                            'lines': lines_count,
                            'complexity': complexity_score,
                            'file_path': rel_path,
                            'file_type': file_ext
                        }
                        
                        # Add node to graph
                        G.add_node(module_name, **module_info[module_name])
                        
                        # Find imports using patterns for this file type
                        for pattern in import_patterns[file_ext]:
                            imports = re.findall(pattern, content, re.MULTILINE)
                            
                            for imported_module in imports:
                                # Clean up the import name
                                imported_module = imported_module.strip()
                                
                                # Skip built-in modules but keep local ones
                                builtin_modules = [
                                    'os', 'sys', 'time', 'datetime', 'json', 'urllib', 're', 'math', 
                                    'collections', 'itertools', 'functools', 'typing', 'pathlib',
                                    'tempfile', 'hashlib', 'pickle', 'shutil', 'random', 'subprocess'
                                ]
                                
                                # Skip relative imports starting with '.' and built-ins
                                if (imported_module.startswith('.') or 
                                    imported_module in builtin_modules or
                                    imported_module.startswith('http') or
                                    imported_module.startswith('std::')):
                                    continue
                                
                                # For Python, handle both local and external imports
                                if file_ext == '.py' and not imported_module.startswith('.'):
                                    # Check if this might be a local module first
                                    potential_local_path = imported_module.replace('.', os.sep) + '.py'
                                    full_potential_path = os.path.join(repo_path, potential_local_path)
                                    
                                    # Also check if it matches any of our discovered modules
                                    is_local_module = (
                                        os.path.exists(full_potential_path) or
                                        imported_module in module_info.keys() or
                                        any(imported_module == local.split('.')[-1] for local in module_info.keys()) or
                                        any(local.endswith(imported_module) for local in module_info.keys())
                                    )
                                    
                                    if is_local_module:
                                        # It's a local module, add the edge
                                        G.add_edge(module_name, imported_module)
                                        print(f"Found local dependency: {module_name} -> {imported_module}")
                                    elif len(imported_module.split('.')) <= 2 and not any(ext in imported_module for ext in ['http', 'www', 'github']):
                                        # It might be an external library, add it but mark differently
                                        G.add_edge(module_name, f"ext:{imported_module}")
                                
                                # For JavaScript/TypeScript, check relative imports
                                elif file_ext in ['.js', '.jsx', '.ts', '.tsx']:
                                    if imported_module.startswith('./') or imported_module.startswith('../'):
                                        # Resolve relative path
                                        import_dir = os.path.dirname(rel_path)
                                        resolved_path = os.path.normpath(os.path.join(import_dir, imported_module))
                                        resolved_module = resolved_path.replace(os.sep, '.').replace('/', '.')
                                        G.add_edge(module_name, resolved_module)
                                    else:
                                        # External module, add but mark as external
                                        if not imported_module.startswith('@') and len(imported_module.split('.')) <= 3:
                                            G.add_edge(module_name, imported_module)
                                
                                # For other languages, add direct dependencies
                                else:
                                    if len(imported_module.split('.')) <= 3:  # Avoid very long module names
                                        G.add_edge(module_name, imported_module)
                        
                    except Exception as e:
                        # Skip files that can't be processed
                        continue
        
        # Keep nodes with low connectivity but remove completely isolated ones
        isolated_nodes = [node for node in G.nodes() if G.degree(node) == 0 and len(G.nodes()) > 5]
        G.remove_nodes_from(isolated_nodes)
        
        # If graph is too large, keep only the most connected components
        if len(G.nodes()) > 50:
            # Get the largest connected component
            if G.number_of_nodes() > 0:
                # For directed graphs, use weakly connected components
                components = list(nx.weakly_connected_components(G))
                if components:
                    largest_component = max(components, key=len)
                    G = G.subgraph(largest_component).copy()
        
        return G
        
        # Debug information
        print(f"Architecture analysis complete: {len(G.nodes())} nodes, {len(G.edges())} edges")
        print(f"Modules found: {list(module_info.keys())}")
        print(f"Edges: {list(G.edges())}")
        
        return G
        
    except Exception as e:
        print(f"Architecture analysis error: {str(e)}")
        st.warning(f"🏗️ **Architecture Diagram Generation Failed**: {str(e)}")
        with st.expander("🔧 Architecture Analysis Tips"):
            st.markdown("""
            **Possible solutions:**
            - **Complex dependencies**: Large projects may have intricate module relationships
            - **File access issues**: Check if all Python files are readable
            - **Import parsing errors**: Some syntax may not be parseable
            - **Memory limitations**: Try analyzing smaller directories first
            """)
        return nx.DiGraph()  # Return empty graph

def serialize_graph_data(G):
    """Convert NetworkX graph to serializable format for session state storage"""
    if len(G.nodes()) == 0:
        return None
    
    try:
        # Convert graph to simple data structure
        graph_data = {
            'nodes': [],
            'edges': []
        }
        
        # Store node data
        for node in G.nodes():
            node_data = G.nodes[node] if hasattr(G.nodes[node], 'get') else {}
            graph_data['nodes'].append({
                'id': node,
                'data': dict(node_data)  # Convert to regular dict
            })
        
        # Store edge data
        for edge in G.edges():
            graph_data['edges'].append({
                'source': edge[0],
                'target': edge[1]
            })
        
        return graph_data
    except Exception as e:
        st.warning(f"⚠️ **Graph Serialization Issue**: Could not serialize architecture data - {str(e)}")
        return None

def deserialize_graph_data(graph_data):
    """Convert serialized graph data back to NetworkX graph"""
    if not graph_data:
        return nx.DiGraph()
    
    try:
        G = nx.DiGraph()
        
        # Add nodes with data
        for node_info in graph_data['nodes']:
            G.add_node(node_info['id'], **node_info['data'])
        
        # Add edges
        for edge_info in graph_data['edges']:
            G.add_edge(edge_info['source'], edge_info['target'])
        
        return G
    except Exception as e:
        st.warning(f"⚠️ **Graph Deserialization Issue**: Could not restore architecture data - {str(e)}")
        return nx.DiGraph()

def analyze_security_vulnerabilities(repo_path):
    """Analyze repository for potential security vulnerabilities and issues"""
    vulnerabilities = []
    improvements = []
    
    try:
        # Security patterns to detect
        security_patterns = {
            'hardcoded_secrets': [
                r'password\s*=\s*["\'][^"\']{3,}["\']',
                r'api_key\s*=\s*["\'][^"\']{10,}["\']',
                r'secret\s*=\s*["\'][^"\']{8,}["\']',
                r'token\s*=\s*["\'][^"\']{10,}["\']',
                r'private_key\s*=\s*["\'][^"\']{20,}["\']'
            ],
            'sql_injection': [
                r'execute\s*\(\s*["\'].*%.*["\']',
                r'query\s*\(\s*["\'].*\+.*["\']',
                r'SELECT.*\+.*FROM',
                r'INSERT.*\+.*VALUES'
            ],
            'weak_crypto': [
                r'md5\s*\(',
                r'sha1\s*\(',
                r'DES\s*\(',
                r'RC4\s*\('
            ],
            'unsafe_eval': [
                r'eval\s*\(',
                r'exec\s*\(',
                r'os\.system\s*\(',
                r'subprocess\.call.*shell\s*=\s*True'
            ],
            'insecure_requests': [
                r'http://.*requests\.',
                r'verify\s*=\s*False',
                r'ssl_verify\s*=\s*False',
                r'InsecureRequestWarning'
            ]
        }
        
        # File extensions to analyze
        code_extensions = ['.py', '.js', '.java', '.php', '.rb', '.go', '.cs', '.cpp', '.c']
        
        security_issues = {}
        file_count = 0
        
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden directories and common build/cache folders
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'build', 'dist']]
            
            for file in files:
                if any(file.endswith(ext) for ext in code_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read().lower()
                            file_count += 1
                            
                            for category, patterns in security_patterns.items():
                                for pattern in patterns:
                                    if re.search(pattern, content, re.IGNORECASE):
                                        if category not in security_issues:
                                            security_issues[category] = []
                                        rel_path = os.path.relpath(file_path, repo_path)
                                        if rel_path not in [item['file'] for item in security_issues[category]]:
                                            security_issues[category].append({
                                                'file': rel_path,
                                                'pattern': pattern
                                            })
                    except:
                        continue
        
        # Convert findings to vulnerabilities
        issue_descriptions = {
            'hardcoded_secrets': {
                'title': '🔐 Hardcoded Secrets Detected',
                'severity': 'critical',
                'description': 'Found potential hardcoded passwords, API keys, or secrets in code',
                'recommendation': 'Move secrets to environment variables or secure config files'
            },
            'sql_injection': {
                'title': '💉 SQL Injection Risk',
                'severity': 'high',
                'description': 'Detected potential SQL injection vulnerabilities',
                'recommendation': 'Use parameterized queries and input validation'
            },
            'weak_crypto': {
                'title': '🔓 Weak Cryptography',
                'severity': 'medium',
                'description': 'Found usage of weak or deprecated cryptographic algorithms',
                'recommendation': 'Upgrade to stronger algorithms like SHA-256, AES-256'
            },
            'unsafe_eval': {
                'title': '⚠️ Unsafe Code Execution',
                'severity': 'high',
                'description': 'Detected potentially unsafe code execution patterns',
                'recommendation': 'Avoid eval(), exec(), and direct shell execution with user input'
            },
            'insecure_requests': {
                'title': '🌐 Insecure Network Communication',
                'severity': 'medium',
                'description': 'Found insecure HTTP requests or disabled SSL verification',
                'recommendation': 'Use HTTPS and enable SSL certificate verification'
            }
        }
        
        for category, issues in security_issues.items():
            if issues and category in issue_descriptions:
                vuln = issue_descriptions[category].copy()
                vuln['count'] = len(issues)
                vuln['files'] = [item['file'] for item in issues[:3]]  # Show first 3 files
                vulnerabilities.append(vuln)
        
        # Code quality improvements
        improvement_patterns = {
            'todo_comments': r'(todo|fixme|hack|xxx)',
            'empty_catch': r'except.*:\s*pass',
            'magic_numbers': r'\b\d{3,}\b',  # Numbers with 3+ digits might be magic numbers
            'long_functions': r'def\s+\w+.*?(?=def|\Z)',  # Basic detection
        }
        
        quality_issues = {}
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__']]
            
            for file in files:
                if file.endswith('.py'):  # Focus on Python for quality analysis
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            
                            # TODO comments
                            if re.search(improvement_patterns['todo_comments'], content, re.IGNORECASE):
                                quality_issues.setdefault('todos', 0)
                                quality_issues['todos'] += len(re.findall(improvement_patterns['todo_comments'], content, re.IGNORECASE))
                            
                            # Empty exception handlers
                            if re.search(improvement_patterns['empty_catch'], content):
                                quality_issues.setdefault('empty_catch', 0)
                                quality_issues['empty_catch'] += 1
                    except:
                        continue
        
        # Add improvement suggestions
        if quality_issues.get('todos', 0) > 5:
            improvements.append({
                'title': '📝 TODO Comments Cleanup',
                'description': f'Found {quality_issues["todos"]} TODO/FIXME comments',
                'recommendation': 'Review and address pending TODO items or create proper issue tickets',
                'priority': 'low'
            })
        
        if quality_issues.get('empty_catch', 0) > 0:
            improvements.append({
                'title': '🐛 Exception Handling',
                'description': f'Found {quality_issues["empty_catch"]} empty exception handlers',
                'recommendation': 'Add proper error logging or handling in catch blocks',
                'priority': 'medium'
            })
        
        return {
            'vulnerabilities': vulnerabilities,
            'improvements': improvements,
            'files_analyzed': file_count
        }
        
    except Exception as e:
        st.warning(f"🔒 **Security Analysis Failed**: {str(e)}")
        with st.expander("🛡️ Security Analysis Tips"):
            st.markdown("""
            **Troubleshooting security scan:**
            - **Large codebase**: Security analysis may timeout on very large repositories  
            - **File access**: Ensure all source files are readable
            - **Binary files**: Some file types may cause parsing issues
            - **Try manual review**: Consider manual security review for critical files
            """)
        return {
            'vulnerabilities': [],
            'improvements': [],
            'files_analyzed': 0,
            'error': str(e)
        }

def display_architecture_visualization(G, repo_name):
    """Display interactive architecture diagram using Plotly with enhanced UI"""
    if len(G.nodes()) == 0:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); 
                    border-radius: 12px; border: 1px solid #e9ecef;">
            <i class="fas fa-search" style="font-size: 2.5rem; color: #34495e; margin-bottom: 1rem;"></i>
            <h3 style="color: #2c3e50;">No Module Dependencies Found</h3>
            <p style="color: #34495e; margin-bottom: 1rem; font-weight: 500;">This could happen if:</p>
            <div style="text-align: left; max-width: 400px; margin: 0 auto;">
                <p style="margin: 0.5rem 0; color: #34495e; font-weight: 500;"><i class="fas fa-circle" style="font-size: 0.5rem; margin-right: 0.5rem;"></i>Repository has no recognizable import/include statements</p>
                <p style="margin: 0.5rem 0; color: #34495e; font-weight: 500;"><i class="fas fa-circle" style="font-size: 0.5rem; margin-right: 0.5rem;"></i>All modules are external dependencies</p>
                <p style="margin: 0.5rem 0; color: #34495e; font-weight: 500;"><i class="fas fa-circle" style="font-size: 0.5rem; margin-right: 0.5rem;"></i>Files couldn't be parsed due to syntax issues</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Compact metrics table
    st.markdown(f"""
    <div style="background: white; border: 1px solid #e1e5e9; border-radius: 4px; padding: 0.75rem; margin: 1rem 0;">
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; text-align: center;">
            <div><strong style="color: #2c3e50;">{len(G.nodes())}</strong><br><small style="color: #34495e; font-weight: 500;">Modules</small></div>
            <div><strong style="color: #2c3e50;">{len(G.edges())}</strong><br><small style="color: #34495e; font-weight: 500;">Dependencies</small></div>
            <div><strong style="color: #2c3e50;">{repo_name}</strong><br><small style="color: #34495e; font-weight: 500;">Repository</small></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        # Use hierarchical layout for better module organization
        if len(G.nodes()) > 50:
            # For large graphs, use circular layout
            pos = nx.circular_layout(G)
        else:
            # For smaller graphs, use spring layout with better parameters
            pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
        
        # Prepare edge traces
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_info.append(f"{edge[0]} → {edge[1]}")
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='rgba(0,0,0,0.8)'),
            hoverinfo='none',
            mode='lines',
            name='Dependencies'
        )
        
        # Prepare node traces
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        node_hover = []
        
        for node in G.nodes():
            if node in pos:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                # Node size based on degree (number of connections)
                degree = G.degree(node)
                size = min(max(degree * 3 + 10, 15), 40)
                node_size.append(size)
                
                # Enhanced node coloring based on role in architecture
                in_degree = G.in_degree(node)
                out_degree = G.out_degree(node)
                total_degree = degree
                
                # More sophisticated coloring scheme
                if in_degree == 0 and out_degree > 0:
                    color = '#ff6b6b'  # Red: Entry points/controllers
                elif out_degree == 0 and in_degree > 0:
                    color = '#4ecdc4'  # Teal: Leaf nodes/utilities
                elif in_degree > out_degree * 2:
                    color = '#45b7d1'  # Blue: Core/shared modules
                elif out_degree > in_degree * 2:
                    color = '#f9ca24'  # Yellow: Heavy dependency users
                elif total_degree > 5:
                    color = '#6c5ce7'  # Purple: Highly connected hubs
                else:
                    color = '#a8e6cf'  # Light green: Balanced modules
                    
                node_color.append(color)
                
                # Truncate long node names for display
                display_name = node.split('.')[-1] if '.' in node else node
                if len(display_name) > 15:
                    display_name = display_name[:12] + '...'
                node_text.append(display_name)
                
                # Hover information
                node_data = G.nodes[node] if hasattr(G.nodes[node], 'get') else {}
                hover_text = f"<b>{node}</b><br>"
                hover_text += f"Connections: {degree}<br>"
                hover_text += f"Dependencies: {out_degree}<br>"
                hover_text += f"Dependents: {in_degree}<br>"
                
                if 'lines' in node_data:
                    hover_text += f"Lines of code: {node_data['lines']}<br>"
                if 'complexity' in node_data:
                    hover_text += f"Complexity score: {node_data['complexity']}<br>"
                if 'file_type' in node_data:
                    hover_text += f"File type: {node_data['file_type']}<br>"
                
                node_hover.append(hover_text)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hovertemplate='%{customdata}<extra></extra>',
            customdata=node_hover,
            text=node_text,
            textposition="middle center",
            textfont=dict(size=9, color='white', family='Inter, sans-serif'),
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white'),
                opacity=0.9,
                sizemode='diameter'
            ),
            name='Modules',
            showlegend=True
        )
        
        # Create the enhanced figure with better styling
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=dict(
                               text=f'🏗️ Interactive Module Dependency Architecture - {repo_name}',
                               x=0.5,
                               font=dict(size=20, color='#000000', family='Inter, sans-serif'),
                               pad=dict(t=20, b=20)
                           ),
                           showlegend=True,
                           hovermode='closest',
                           margin=dict(b=40,l=20,r=20,t=80),
                           annotations=[ 
                               dict(
                                   text="🎯 Interactive Dependency Graph<br>🔵 Core Modules  🔴 High Dependencies  🟢 Balanced  🟡 Leaf Modules<br>Click nodes • Drag to explore • Hover for details",
                                   showarrow=False,
                                   xref="paper", yref="paper",
                                   x=0.005, y=-0.02,
                                   xanchor="left", yanchor="bottom",
                                   font=dict(color="#000000", size=14, family='Inter, sans-serif'),
                                   bgcolor="rgba(255,255,255,0.95)",
                                   bordercolor="#000000",
                                   borderwidth=2,
                                   borderpad=10
                               )
                           ],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='#fafafa',
                           paper_bgcolor='white',
                           font=dict(family='Inter, sans-serif', color='#000000'),
                           legend=dict(
                               font=dict(color='#000000', size=12),
                               bgcolor='rgba(255,255,255,0.9)',
                               bordercolor='#000000',
                               borderwidth=1
                           )
                       )
                      )
        
        # Display the interactive chart
        st.plotly_chart(fig, use_container_width=True, key=f"architecture_viz_{repo_name}")
        
        # Enhanced architectural metrics focusing on meaningful insights
        total_modules = len(G.nodes())
        total_dependencies = len(G.edges())
        
        # Find key architectural elements
        entry_points = [node for node in G.nodes() if G.in_degree(node) == 0 and G.out_degree(node) > 0]
        core_modules = [node for node in G.nodes() if G.in_degree(node) > 3]
        leaf_modules = [node for node in G.nodes() if G.out_degree(node) == 0 and G.in_degree(node) > 0]
        hub_modules = [node for node in G.nodes() if G.degree(node) > 5]
        
        # Most connected module
        most_connected = max(G.nodes(), key=lambda x: G.degree(x)) if total_modules > 0 else "None"
        most_connected_degree = G.degree(most_connected) if total_modules > 0 else 0
        display_name = most_connected.split('.')[-1] if '.' in most_connected else most_connected
        if len(display_name) > 12:
            display_name = display_name[:10] + '..'
        
        # Compact architectural insights
        if len(G.nodes()) > 5:
            st.markdown("### Architectural Analysis")
            
            # Find insights
            insights = []
            degrees = [G.degree(node) for node in G.nodes()]
            avg_degree_local = sum(degrees) / len(degrees) if degrees else 0
            
            # High connectivity check
            high_degree_nodes = [node for node in G.nodes() if G.degree(node) > avg_degree_local * 2]
            if high_degree_nodes:
                insights.append({
                    'type': 'warning',
                    'title': 'Highly Connected Modules',
                    'description': f"Found {len(high_degree_nodes)} modules with high connectivity",
                    'recommendation': "Consider breaking down these modules to reduce coupling"
                })
            
            # Isolated modules check
            isolated_nodes = [node for node in G.nodes() if G.degree(node) == 0]
            if isolated_nodes:
                insights.append({
                    'type': 'info',
                    'title': 'Isolated Modules',
                    'description': f"Found {len(isolated_nodes)} isolated modules with no dependencies",
                    'recommendation': "Review if these modules are needed or should be connected"
                })
            
            # Display insights
            for insight in insights:
                if insight['type'] == 'warning':
                    st.warning(f"**{insight['title']}**: {insight['description']} - {insight['recommendation']}")
                else:
                    st.info(f"**{insight['title']}**: {insight['description']} - {insight['recommendation']}")
    
    except Exception as e:
        st.error(f"Error generating architecture visualization: {str(e)}")


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
    
    # === ENHANCED: Interactive Architecture Visualization ===
    st.markdown("""
    <div class="architecture-section">
        <div class="architecture-header">
            <i class="fas fa-project-diagram architecture-icon"></i>
            <h2 class="architecture-title">Interactive Architecture Visualization</h2>
        </div>
        <p style="color: #34495e; margin-bottom: 1.5rem; font-size: 1.1rem; font-weight: 500;">
            Explore your codebase structure with an interactive dependency graph showing module relationships, 
            architectural patterns, and potential optimization opportunities.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if architecture graph data was pre-generated during analysis
    if 'architecture_graph_data' in metrics and metrics['architecture_graph_data']:
        try:
            dependency_graph = deserialize_graph_data(metrics['architecture_graph_data'])
            if dependency_graph and dependency_graph.number_of_nodes() > 0:
                # Display the interactive visualization with enhanced UI
                st.success(f"✅ **Architecture Analysis Complete** - Found {dependency_graph.number_of_nodes()} modules with {dependency_graph.number_of_edges()} dependencies")
                display_architecture_visualization(dependency_graph, repo_name)
            else:
                st.info("🔍 **No module dependencies found** - Repository may not have complex interconnections or supported file types.")
        except Exception as e:
            st.error(f"❌ **Architecture visualization error**: {str(e)}")
    elif 'architecture_graph_data' in metrics and metrics['architecture_graph_data'] is None:
        st.warning("⚠️ **Architecture analysis completed but no dependencies detected** - This repository may use external dependencies or have simple structure.")
    else:
        # Try to generate architecture on-the-fly if not available
        if st.button("🏗️ Generate Architecture Visualization", type="secondary"):
            with st.spinner("🔄 Analyzing architecture and dependencies..."):
                try:
                    # Get the repository path from cache or clone again
                    # Try to get repo_url from session state or reconstruct cache path
                    if 'current_repo_url' in st.session_state:
                        repo_url = st.session_state.current_repo_url
                        cache_key = hashlib.md5(repo_url.encode()).hexdigest()
                        cache_path = os.path.join(CACHE_DIR, cache_key)
                    else:
                        st.warning("⚠️ Repository URL not found. Please run 'Analyze Repository Metrics' first.")
                        return
                    
                    if os.path.exists(cache_path):
                        dependency_graph = generate_architecture_diagram(cache_path)
                        if dependency_graph and dependency_graph.number_of_nodes() > 0:
                            metrics['architecture_graph_data'] = serialize_graph_data(dependency_graph)
                            st.session_state[f'metrics_{repo_name}'] = metrics
                            st.success("✅ Architecture analysis complete!")
                            st.rerun()
                        else:
                            st.info("🔍 **No module dependencies found** - Repository may not have complex interconnections.")
                    else:
                        st.warning("⚠️ Repository data not available. Please run 'Analyze Repository Metrics' first.")
                except Exception as e:
                    st.error(f"❌ **Architecture generation failed**: {str(e)}")
        else:
            st.markdown("""
            <div class="architecture-section">
                <div style="text-align: center; padding: 2rem;">
                    <i class="fas fa-search" style="font-size: 3rem; color: #667eea; margin-bottom: 1rem;"></i>
                    <h3 style="color: #000000; margin-bottom: 1rem;">Architecture Analysis Not Available</h3>
                    <p style="color: #000000; margin-bottom: 1.5rem; font-weight: 500;">
                        Click "🔍 Analyze Repository Metrics" to generate the interactive architecture diagram,<br>
                        or use the button above to generate it now.
                    </p>
                    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                                padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                        <p style="margin: 0; color: #000000; font-size: 0.9rem;">
                            <i class="fas fa-info-circle"></i>
                            The architecture visualization analyzes module dependencies across multiple programming languages
                            and provides insights into code structure, coupling, and architectural patterns.
                        </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # === Security Vulnerabilities & Code Quality Analysis ===
    st.markdown("""
    <div class="architecture-section">
        <div class="architecture-header">
            <span class="architecture-icon">🔒</span>
            <h2 class="architecture-title">Security & Code Quality Analysis</h2>
        </div>
        <p style="color: #34495e; margin-bottom: 1rem; font-size: 1rem; font-weight: 500;">
            Security vulnerability scanning and code quality assessment.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display security analysis if available
    if 'security_analysis' in metrics and metrics['security_analysis']:
        security_data = metrics['security_analysis']
        vulnerabilities = security_data.get('vulnerabilities', [])
        improvements = security_data.get('improvements', [])
        files_analyzed = security_data.get('files_analyzed', 0)
        
        # Security overview metrics
        critical_count = len([v for v in vulnerabilities if v.get('severity') == 'critical'])
        high_count = len([v for v in vulnerabilities if v.get('severity') == 'high'])
        medium_count = len([v for v in vulnerabilities if v.get('severity') == 'medium'])
        
        st.markdown(f"""
        <div style="background: white; border: 1px solid #e1e5e9; border-radius: 4px; padding: 0.75rem; margin: 1rem 0;">
            <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 1rem; text-align: center;">
                <div><strong>{files_analyzed}</strong><br><small style="color: #34495e; font-weight: 500;">Files</small></div>
                <div><strong style="color: #dc3545;">{critical_count}</strong><br><small style="color: #34495e; font-weight: 500;">Critical</small></div>
                <div><strong style="color: #fd7e14;">{high_count}</strong><br><small style="color: #34495e; font-weight: 500;">High</small></div>
                <div><strong style="color: #ffc107;">{medium_count}</strong><br><small style="color: #34495e; font-weight: 500;">Medium</small></div>
                <div><strong style="color: #007bff;">{len(improvements)}</strong><br><small style="color: #34495e; font-weight: 500;">Improvements</small></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display vulnerabilities
        if vulnerabilities:
            st.markdown("### ⚠️ Security Vulnerabilities Found")
            
            for vuln in vulnerabilities:
                severity_colors = {
                    'critical': '#dc3545',
                    'high': '#fd7e14', 
                    'medium': '#ffc107'
                }
                
                color = severity_colors.get(vuln.get('severity', 'medium'), '#6c757d')
                
                files_text = f"Found in {vuln.get('count', 0)} locations"
                if vuln.get('files'):
                    files_text += f": {', '.join(vuln['files'][:2])}"
                    if vuln.get('count', 0) > 2:
                        files_text += f" and {vuln.get('count', 0) - 2} more files"
                
                st.markdown(f"""
                <div style="background: white; border-left: 3px solid {color}; padding: 1rem; 
                            margin: 1rem 0; border-radius: 4px; border: 1px solid #e1e5e9;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #2c3e50;">{vuln.get('title', 'Security Issue')}</h4>
                    <p style="margin: 0 0 0.5rem 0; color: #34495e; font-weight: 500;">{vuln.get('description', '')}</p>
                    <p style="margin: 0 0 0.5rem 0; color: #34495e; font-size: 0.9rem; font-weight: 500;">{files_text}</p>
                    <p style="margin: 0; color: #007bff; font-size: 0.9rem;">
                        💡 {vuln.get('recommendation', '')}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        # Display improvements
        if improvements:
            st.markdown("### 💡 Code Quality Improvements")
            
            for improvement in improvements:
                priority_colors = {
                    'high': '#fd7e14',
                    'medium': '#ffc107',
                    'low': '#28a745'
                }
                
                color = priority_colors.get(improvement.get('priority', 'low'), '#28a745')
                
                st.markdown(f"""
                <div style="background: white; border-left: 3px solid {color}; padding: 1rem; 
                            margin: 1rem 0; border-radius: 4px; border: 1px solid #e1e5e9;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #2c3e50;">{improvement.get('title', 'Improvement')}</h4>
                    <p style="margin: 0 0 0.5rem 0; color: #34495e; font-weight: 500;">{improvement.get('description', '')}</p>
                    <p style="margin: 0; color: #007bff; font-size: 0.9rem;">
                        💡 {improvement.get('recommendation', '')}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        # No issues found
        if not vulnerabilities and not improvements:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background: #d4edda; 
                        border-radius: 6px; border: 1px solid #28a745; margin: 1rem 0;">
                <h3 style="color: #155724;">✅ No Security Issues Detected!</h3>
                <p style="color: #155724; margin-bottom: 0;">
                    No obvious security vulnerabilities or code quality issues were found.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: #f8f9fa; 
                    border-radius: 6px; border: 1px solid #e1e5e9;">
            <h3 style="color: #2c3e50;">🔍 Security Analysis Not Available</h3>
            <p style="color: #34495e; font-weight: 500;">
                Click "Analyze Repository Metrics" to run security and code quality analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)

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

            # Updated to use new RunnableSequence pattern instead of deprecated LLMChain
            llm_chain = prompt | llm

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
    # Apply modern styling first
    apply_modern_styling()
    
    # Simple Header Section
    st.markdown("""
    <div class="modern-header">
        <h1 class="header-title">Smart Repository Analyzer</h1>
        <p class="header-subtitle">AI-Powered Repository Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add cache management in sidebar with modern styling
    with st.sidebar:
        st.markdown('<h3 style="color: #ffffff;">📁 Cache Management</h3>', unsafe_allow_html=True)
        if st.button("🗑️ Clear All Cache"):
            if os.path.exists(CACHE_DIR):
                shutil.rmtree(CACHE_DIR)
                os.makedirs(CACHE_DIR)
            st.session_state.cached_repos = {}
            st.success("Cache cleared!")
        
        # Show cache status
        cache_count = len(os.listdir(CACHE_DIR)) if os.path.exists(CACHE_DIR) else 0
        st.markdown(f'<p style="color: #ffffff;">📊 Cached repositories: {cache_count}</p>', unsafe_allow_html=True)
    
    
    # Modern Repository Input Section
    st.markdown("### Repository Analysis")
    st.markdown("Enter a GitHub repository URL to start analysis.")
    
    repo_url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/username/repository-name",
        key="repo_url_input",
        help="Enter the complete GitHub URL"
    )
    
    # Reset conversation if repo changes
    if repo_url != st.session_state.current_repo:
        st.session_state.conversation_count = 0
        st.session_state.conversation_history = ""
        st.session_state.qa_history = []
        st.session_state.current_repo = repo_url
        st.session_state.current_question_context = None  # Reset cached context
    
    if not repo_url:  # Skip if no URL is provided
        st.info("Enter a GitHub repository URL above to get started")
        return
    
    # Add sub-navigation tabs after repo URL input
    tab1, tab2 = st.tabs(["Repository Metrics", "AI Chat Analysis"])
    
    repo_name = repo_url.split("/")[-1]
    
    # Clear old cache files periodically
    clear_old_cache()
    
    with tab1:
        st.header("📊 Repository Analytics Dashboard")
        
        if st.button("🔍 Analyze Repository Metrics", type="primary"):
            # Store the current repo URL for later use
            st.session_state.current_repo_url = repo_url
            with st.spinner("🔄 Cloning and analyzing repository..."):
                with tempfile.TemporaryDirectory() as local_path:
                    if clone_git_repo(repo_url, local_path):
                        metrics = analyze_repository_metrics(local_path)
                        if metrics:
                            # Generate architecture diagram while repo is still available
                            with st.spinner("🏗️ Analyzing architecture and dependencies..."):
                                dependency_graph = generate_architecture_diagram(local_path)
                                if dependency_graph and dependency_graph.number_of_nodes() > 0:
                                    print(f"Architecture graph generated: {dependency_graph.number_of_nodes()} nodes, {dependency_graph.number_of_edges()} edges")
                                    # Store the graph data in serializable format
                                    metrics['architecture_graph_data'] = serialize_graph_data(dependency_graph)
                                else:
                                    print("No architecture graph generated - no dependencies found")
                                    metrics['architecture_graph_data'] = None
                            
                            # Analyze security vulnerabilities and code quality
                            with st.spinner("🔒 Scanning for security vulnerabilities and code quality issues..."):
                                security_analysis = analyze_security_vulnerabilities(local_path)
                                metrics['security_analysis'] = security_analysis
                                
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

                # Updated to use new RunnableSequence pattern instead of deprecated LLMChain
                llm_chain = prompt | llm

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

        # Dynamic question input key to prevent state conflicts
        question_key = f"user_question_input_{st.session_state.conversation_count}"
        user_question = st.text_input(
            "Ask a question about the repository:",
            key=question_key,
            help="Type your question and press Enter or click Submit"
        )
        
        # Simple submit button
        submit_clicked = st.button("Submit Question", type="primary")

        # Initialize qa_history if it doesn't exist
        if 'qa_history' not in st.session_state:
            st.session_state.qa_history = []

        # Display conversation history first (if exists)
        if st.session_state.qa_history:
            st.subheader("💬 Conversation History")
            for i, (q, a) in enumerate(st.session_state.qa_history):
                with st.expander(f"Q{i+1}: {q[:80]}{'...' if len(q) > 80 else ''}", expanded=False):
                    st.write(f"**Question:** {q}")
                    st.write(f"**Answer:**")
                    display_enhanced_answer(a)
            st.divider()

        # Process question when submit is clicked
        if submit_clicked and user_question.strip():
            if user_question.lower().strip() == "exit()":
                st.warning("Session ended")
                return
            try:
                with st.spinner("🤖 Processing your question..."):
                    # Ensure question_context exists
                    if 'question_context' not in locals() or question_context is None:
                        st.error("❌ Repository data not loaded. Please ensure the repository was processed successfully.")
                        return
                    
                    # Format the question
                    formatted_question = format_questions(user_question)
                    
                    # Update conversation history in context
                    question_context.conversation_history = st.session_state.conversation_history
                    
                    # Get the answer
                    answer = ask_question(formatted_question, question_context)
                    
                    # Add to QA history
                    st.session_state.qa_history.append((user_question, answer))
                    
                    # Update conversation history for context
                    formatted_history = ""
                    for i, (q, a) in enumerate(st.session_state.qa_history):
                        formatted_history += f"===== Previous Conversation {i+1} =====\n"
                        formatted_history += f"User: {q}\n"
                        formatted_history += f"Assistant: {a}\n\n"
                    
                    st.session_state.conversation_history = formatted_history
                    st.session_state.conversation_count += 1
                    
                    # Display the answer immediately
                    st.success("✅ **Latest Answer:**")
                    display_enhanced_answer(answer)
                    
                    # Rerun to refresh the interface (input will be cleared automatically)
                    st.rerun()
                    
            except Exception as ex:
                error_message = str(ex)
                print(f"Chat Error: {ex}")  # Log to console for debugging
                
                # User-friendly error messages
                if "503" in error_message or "Service unavailable" in error_message:
                    st.error("🔄 **AI service temporarily unavailable.** Please try again in a moment.")
                elif "rate limit" in error_message.lower():
                    st.error("⏰ **Rate limit reached.** Please wait before asking another question.")
                elif "authentication" in error_message.lower() or "api_key" in error_message.lower():
                    st.error("🔑 **Authentication issue.** Please check your GROQ_API_KEY in environment variables.")
                elif "timeout" in error_message.lower():
                    st.error("⏱️ **Request timed out.** Please try with a shorter question.")
                else:
                    st.error(f"❌ **Error occurred:** {error_message}")
                    with st.expander("🔧 Troubleshooting"):
                        st.write("**Possible solutions:**")
                        st.write("- Check your internet connection")
                        st.write("- Verify GROQ_API_KEY is set correctly")
                        st.write("- Try a simpler question")
                        st.write("- Reload the page and try again")
                
                st.session_state.conversation_count += 1

        elif submit_clicked and not user_question.strip():
            st.warning("⚠️ Please enter a question before submitting.")
        
        # Show helpful prompts if no conversation yet
        if not st.session_state.qa_history:
            st.info("💡 **Try asking questions like:**")
            col1, col2 = st.columns(2)
            with col1:
                st.write("• What is this repository about?")
                st.write("• What programming languages are used?")
                st.write("• How is the code structured?")
            with col2:
                st.write("• What are the main components?")
                st.write("• Are there any security issues?")
                st.write("• How can I contribute to this project?")

if __name__ == "__main__":
    main()