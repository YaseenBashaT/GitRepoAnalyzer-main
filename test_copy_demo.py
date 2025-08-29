#!/usr/bin/env python3
"""
Test script to demonstrate the new copy functionality
"""

import streamlit as st
import re

def test_copy_functionality():
    """Demo the copy functionality with sample responses"""
    
    sample_response_with_code = '''
This is a sample response with code blocks that you can now copy easily!

Here's a Python function:

```python
def hello_world():
    print("Hello, World!")
    return "Hello from Python"

# Call the function
result = hello_world()
print(f"Result: {result}")
```

And here's a bash command:

```bash
# Install dependencies
pip install streamlit
pip install langchain

# Run the application
streamlit run main.py --server.port=8506
```

You can also have configuration files:

```yaml
# config.yaml
app:
  name: "GitRepoAnalyzer"
  version: "2.0"
  features:
    - "Code Analysis"
    - "Copy Functionality"
    - "Caching System"
```

Each code block now has its own copy button! 🎉
'''

    print("🧪 Testing Copy Functionality")
    print("=" * 50)
    print("Sample response with multiple code blocks:")
    print(sample_response_with_code)
    print("\n✅ Copy functionality added to GitRepoAnalyzer!")
    print("Features:")
    print("📋 Individual copy buttons for each code block")
    print("📋 Copy All Code button for multiple blocks") 
    print("📋 Copy entire response button for text")
    print("🎨 Enhanced formatting with language tags")
    print("🔧 Proper syntax highlighting")

if __name__ == "__main__":
    test_copy_functionality()
