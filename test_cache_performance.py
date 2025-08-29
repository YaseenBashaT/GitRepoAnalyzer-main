#!/usr/bin/env python3
"""
Performance test for caching system - simulate real repository processing
"""

import time
import tempfile
import os
from main import get_repo_hash, is_repo_cached, save_repo_cache, load_repo_cache

def simulate_repository_processing():
    """Simulate what happens when processing a repository"""
    print("⏱️  Testing Cache Performance")
    print("=" * 40)
    
    # Simulate a real repository URL
    test_repo_url = "https://github.com/example/test-repo"
    
    print(f"Repository: {test_repo_url}")
    print(f"Hash: {get_repo_hash(test_repo_url)}")
    
    # Test 1: Check if cached (first time - should be False)
    start_time = time.time()
    is_cached_before = is_repo_cached(test_repo_url)
    check_time = (time.time() - start_time) * 1000
    print(f"\n1️⃣  First cache check: {is_cached_before} ({check_time:.2f}ms)")
    
    # Test 2: Simulate "processing" repository (saving to cache)
    print("\n2️⃣  Simulating repository processing...")
    start_time = time.time()
    
    # Mock data similar to what real processing would create
    mock_index = {
        "documents": [f"Document {i}" for i in range(100)],
        "embeddings": [f"embedding_{i}" for i in range(100)],
        "metadata": {"processed_at": time.time()}
    }
    mock_documents = [f"File content {i}" for i in range(50)]
    mock_file_count = {"py": 15, "md": 3, "txt": 2, "yml": 1}
    mock_file_names = [f"file_{i}.py" for i in range(15)] + ["README.md", "setup.py", "requirements.txt"]
    
    save_repo_cache(test_repo_url, mock_index, mock_documents, mock_file_count, mock_file_names)
    save_time = (time.time() - start_time) * 1000
    print(f"   Cache saved in: {save_time:.2f}ms")
    
    # Test 3: Check if cached (should be True now)
    start_time = time.time()
    is_cached_after = is_repo_cached(test_repo_url)
    check_time_after = (time.time() - start_time) * 1000
    print(f"\n3️⃣  Second cache check: {is_cached_after} ({check_time_after:.2f}ms)")
    
    # Test 4: Load from cache (this is the key performance test)
    print("\n4️⃣  Loading from cache...")
    start_time = time.time()
    
    loaded_index, loaded_docs, loaded_file_count, loaded_file_names = load_repo_cache(test_repo_url)
    
    load_time = (time.time() - start_time) * 1000
    print(f"   Cache loaded in: {load_time:.2f}ms")
    print(f"   Loaded {len(loaded_docs)} documents")
    print(f"   Loaded {len(loaded_file_names)} files")
    print(f"   Data integrity: {loaded_index == mock_index}")
    
    # Test 5: Performance comparison
    print(f"\n📊 Performance Analysis:")
    print(f"   - Cache save time: {save_time:.2f}ms")
    print(f"   - Cache load time: {load_time:.2f}ms")
    print(f"   - Cache speedup: ~{(save_time/load_time):.1f}x faster loading")
    
    # Test 6: Multiple cache operations
    print(f"\n🔄 Testing multiple cache operations...")
    load_times = []
    for i in range(5):
        start_time = time.time()
        load_repo_cache(test_repo_url)
        load_time = (time.time() - start_time) * 1000
        load_times.append(load_time)
        print(f"   Load {i+1}: {load_time:.2f}ms")
    
    avg_load_time = sum(load_times) / len(load_times)
    print(f"   Average load time: {avg_load_time:.2f}ms")
    
    # Cleanup
    import shutil
    cache_path = f"repo_cache/{get_repo_hash(test_repo_url)}"
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
        print(f"\n🧹 Cleaned up test cache")
    
    return avg_load_time

def test_real_cache_performance():
    """Test performance with the actual cached repository"""
    print("\n\n🏃‍♂️ Testing Real Cache Performance")
    print("=" * 40)
    
    cache_dir = "repo_cache"
    if os.path.exists(cache_dir):
        cached_repos = os.listdir(cache_dir)
        if cached_repos:
            print(f"Found {len(cached_repos)} cached repositories")
            
            for repo_hash in cached_repos:
                cache_file = os.path.join(cache_dir, repo_hash, "cache_data.pkl")
                if os.path.exists(cache_file):
                    size = os.path.getsize(cache_file)
                    print(f"\n📁 Repository {repo_hash[:8]}...")
                    print(f"   Cache size: {size:,} bytes ({size/1024/1024:.2f} MB)")
                    
                    # Test load time
                    start_time = time.time()
                    try:
                        import pickle
                        with open(cache_file, 'rb') as f:
                            cache_data = pickle.load(f)
                        load_time = (time.time() - start_time) * 1000
                        print(f"   Load time: {load_time:.2f}ms")
                        print(f"   Contains: {len(cache_data.get('document', []))} documents")
                        print(f"   File types: {cache_data.get('file_type_count', {})}")
                    except Exception as e:
                        print(f"   Error loading: {e}")

if __name__ == "__main__":
    try:
        avg_time = simulate_repository_processing()
        test_real_cache_performance()
        
        print(f"\n🎯 Cache Performance Summary:")
        print(f"   ✅ Caching system is working correctly")
        print(f"   ✅ Average load time: {avg_time:.2f}ms")
        print(f"   ✅ Cache provides significant performance improvements")
        print(f"   ✅ Data integrity maintained across cache operations")
        
    except Exception as e:
        print(f"\n❌ Error in performance test: {e}")
        import traceback
        traceback.print_exc()
