from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment
api_key = os.getenv("GROQ_API_KEY")

try:
    # Initialize Groq client
    client = Groq(api_key=api_key)
    
    # Test API with a simple request
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": "Say hello!"}]
    )
    
    # Print response
    print("API Test Successful!")
    print("Response:", response.choices[0].message.content)
    
except Exception as e:
    print("Error occurred:")
    print(e)
