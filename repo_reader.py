import os
import subprocess
from rank_bm25 import BM25Okapi
from langchain_community.document_loaders import DirectoryLoader, NotebookLoader
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utility import clean_and_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clone_git_repo(url, path):
    try:
        subprocess.run(['git','clone',url,path],check=True)
        return True
    except subprocess.CalledProcessError as ex:
        print(f"failed to clone repo: {ex}")
        return False
    
def load_and_index_files(repo_path):
    import glob as glob_module
    from langchain_core.documents import Document
    
    # Define file extensions we want to process
    extensions = ['txt', 'md', 'markdown', 'rst', 'py', 'js', 'ts', 'jsx', 'tsx', 'java', 'c', 'cpp', 'cs', 'go', 'rb', 'php', 'scala', 'html', 'htm', 'xml', 'json', 'yaml', 'yml', 'ini', 'toml', 'cfg', 'conf', 'sh', 'bash', 'css', 'scss', 'sql', 'vue', 'svelte', 'r', 'R', 'dart', 'kt', 'swift', 'pl', 'lua']
    
    file_type_counts = {}
    documents_dict = {}
    total_processed = 0
    total_errors = 0
    
    def load_file_content(file_path):
        """Load content from a single file with robust error handling"""
        try:
            # Skip very large files
            if os.path.getsize(file_path) > 10 * 1024 * 1024:  # 10MB limit
                return None
            
            # Try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                        content = f.read()
                    break
                except (UnicodeDecodeError, UnicodeError, PermissionError):
                    continue
                except Exception:
                    continue
            
            if content is None:
                return None
            
            # Skip empty or very short files
            if len(content.strip()) < 5:
                return None
            
            # Basic binary detection - skip files with too many null bytes or non-printable chars
            if '\x00' in content[:1000] or content.count('\x00') > 10:
                return None
            
            # Check if file appears to be text (reasonable ratio of printable characters)
            sample = content[:2000]  # Check first 2KB
            if sample:
                printable_count = sum(1 for c in sample if c.isprintable() or c in '\n\r\t')
                if printable_count / len(sample) < 0.7:  # Less than 70% printable = likely binary
                    return None
            
            return content
            
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None
    
    # Process each file extension
    for ext in extensions:
        ext_file_count = 0
        
        # Find all files with this extension
        pattern = os.path.join(repo_path, '**', f'*.{ext}')
        matching_files = glob_module.glob(pattern, recursive=True)
        
        for file_path in matching_files:
            try:
                # Skip hidden files and directories
                if any(part.startswith('.') for part in os.path.relpath(file_path, repo_path).split(os.sep)):
                    if not any(file_path.endswith(f'.{allowed}') for allowed in ['gitignore', 'dockerignore', 'editorconfig']):
                        continue
                
                # Load file content
                content = load_file_content(file_path)
                if content is not None:
                    # Create document
                    relative_path = os.path.relpath(file_path, repo_path)
                    file_id = str(uuid.uuid4())
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": relative_path,
                            "file_id": file_id
                        }
                    )
                    
                    documents_dict[file_id] = doc
                    ext_file_count += 1
                    total_processed += 1
                else:
                    total_errors += 1
                    
            except Exception as e:
                total_errors += 1
                continue
        
        if ext_file_count > 0:
            file_type_counts[ext] = ext_file_count
    
    print(f"Repository indexing complete: {total_processed} files processed, {total_errors} errors")
    print(f"File types found: {list(file_type_counts.keys())}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)

    split_documents = []
    for file_id, original_doc in documents_dict.items():
        split_docs = text_splitter.split_documents([original_doc])
        for split_doc in split_docs:
            split_doc.metadata['file_id'] = original_doc.metadata['file_id']
            split_doc.metadata['source'] = original_doc.metadata['source']

        split_documents.extend(split_docs)

    index = None
    if split_documents:
        tokenized_documents = [clean_and_tokenize(doc.page_content) for doc in split_documents]
        index = BM25Okapi(tokenized_documents)
    return index, split_documents, file_type_counts, [doc.metadata['source'] for doc in split_documents]

def search_documents(query, index, documents, n_results=5):
    query_tokens = clean_and_tokenize(query)
    bm25_scores = index.get_scores(query_tokens)

    tfidf_vectorizer = TfidfVectorizer(tokenizer=clean_and_tokenize, lowercase=True, stop_words='english', use_idf=True, smooth_idf=True, sublinear_tf=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform([doc.page_content for doc in documents])
    query_tfidf = tfidf_vectorizer.transform([query])

    cosine_sim_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

    combined_scores = bm25_scores * 0.5 + cosine_sim_scores * 0.5

    unique_top_document_indices = list(set(combined_scores.argsort()[::-1]))[:n_results]

    return [documents[i] for i in unique_top_document_indices]
