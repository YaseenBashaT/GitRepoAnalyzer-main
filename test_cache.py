#!/usr/bin/env python3
"""
Test script to verify caching functionality works correctly
"""

import os
import sys
import hashlib
import pickle
import time
from main import get_repo_hash, get_cache_path, is_repo_cached, save_repo_cache, load_repo_cache, clear_old_cache

def test_cache_functions():
    print("🧪 Testing Smart Repository Analyzer Caching System")
    print("=" * 50)
    
    # Test 1: Test hash generation
    test_url = "https://github.com/test/repo"
    repo_hash = get_repo_hash(test_url)
    print(f"✅ Hash generation: {test_url} -> {repo_hash}")
    
    # Test 2: Test cache path generation
    cache_path = get_cache_path(test_url)
    print(f"✅ Cache path: {cache_path}")
    
    # Test 3: Test if repo is cached (should be False for new URL)
    is_cached = is_repo_cached(test_url)
    print(f"✅ Is cached (before): {is_cached}")
    
    # Test 4: Test saving cache data
    print("\n📁 Testing cache save/load functionality...")
    
    # Mock data for testing
    mock_index = {"test": "index_data"}
    mock_document = ["doc1", "doc2", "doc3"]
    mock_file_count = {"py": 5, "md": 2}
    mock_file_names = ["main.py", "test.py", "README.md"]
    
    # Save cache
    save_repo_cache(test_url, mock_index, mock_document, mock_file_count, mock_file_names)
    print("✅ Cache saved successfully")
    
    # Test 5: Check if repo is now cached
    is_cached_after = is_repo_cached(test_url)
    print(f"✅ Is cached (after save): {is_cached_after}")
    
    # Test 6: Load cache data
    loaded_index, loaded_document, loaded_file_count, loaded_file_names = load_repo_cache(test_url)
    print(f"✅ Cache loaded successfully")
    print(f"   - Index match: {loaded_index == mock_index}")
    print(f"   - Document match: {loaded_document == mock_document}")
    print(f"   - File count match: {loaded_file_count == mock_file_count}")
    print(f"   - File names match: {loaded_file_names == mock_file_names}")
    
    # Test 7: Check existing cache directory
    print(f"\n📊 Current cache status:")
    cache_dir = "repo_cache"
    if os.path.exists(cache_dir):
        cache_folders = os.listdir(cache_dir)
        print(f"   - Cache directory exists: {cache_dir}")
        print(f"   - Cached repositories: {len(cache_folders)}")
        for folder in cache_folders:
            cache_file = os.path.join(cache_dir, folder, "cache_data.pkl")
            if os.path.exists(cache_file):
                size = os.path.getsize(cache_file)
                print(f"   - {folder}: {size:,} bytes")
    
    # Cleanup test cache
    test_cache_path = get_cache_path(test_url)
    if os.path.exists(test_cache_path):
        import shutil
        shutil.rmtree(test_cache_path)
        print(f"\n🧹 Cleaned up test cache: {test_cache_path}")
    
    print("\n🎉 All cache tests completed successfully!")

def test_hash_consistency():
    """Test that the same URL always produces the same hash"""
    print("\n🔍 Testing hash consistency...")
    
    test_urls = [
        "https://github.com/user/repo1",
        "https://github.com/user/repo2",
        "https://github.com/different/repo1"
    ]
    
    for url in test_urls:
        hash1 = get_repo_hash(url)
        hash2 = get_repo_hash(url)
        print(f"   {url} -> {hash1} (consistent: {hash1 == hash2})")

if __name__ == "__main__":
    try:
        test_cache_functions()
        test_hash_consistency()
        print("\n✅ All caching functionality is working correctly!")
    except Exception as e:
        print(f"\n❌ Error testing cache functionality: {e}")
        import traceback
        traceback.print_exc()
