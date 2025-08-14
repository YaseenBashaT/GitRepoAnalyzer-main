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

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
model_name = "llama3-70b-8192"  # Latest available model from Groq

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
                            st.warning(f"Groq service temporarily unavailable. Retrying in {wait_time:.1f} seconds... (Attempt {attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                            continue
                        else:
                            # Fallback response for final failure
                            fallback_response = f"I apologize, but the AI service is currently experiencing issues. However, I can see that your repository has been successfully cloned and indexed. You can try asking your question again in a moment, or here's what I can tell you from the repository structure: The repository appears to contain code files that have been successfully processed. Please try your question again shortly when the service is restored."
                            generations.append([Generation(text=fallback_response)])
                            st.error("AI service temporarily unavailable. Using fallback response.")
                            break
                    
                    elif "rate limit" in error_message.lower():
                        if attempt < max_retries - 1:
                            wait_time = 60  # Wait longer for rate limits
                            st.warning(f"Rate limit reached. Waiting {wait_time} seconds before retry...")
                            time.sleep(wait_time)
                            continue
                        else:
                            fallback_response = "Rate limit exceeded. Please wait a moment before asking another question."
                            generations.append([Generation(text=fallback_response)])
                            break
                    
                    else:
                        # Other errors
                        if attempt < max_retries - 1:
                            st.warning(f"Error occurred: {error_message}. Retrying... (Attempt {attempt + 1}/{max_retries})")
                            time.sleep(retry_delay)
                            continue
                        else:
                            fallback_response = f"I encountered an error processing your request: {error_message}. Please try rephrasing your question or try again later."
                            generations.append([Generation(text=fallback_response)])
                            break
        
        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "groq"
def main():
    st.title("🚀 Advanced Git Repository Analyzer")
    st.markdown("### AI-Powered Code Analysis with Real-Time Metrics Dashboard")
    
    # Add info about common warnings
    with st.expander("ℹ️ About File Loading Warnings", expanded=False):
        st.markdown("""
        **Common warnings you might see (these are normal):**
        - `Error loading files with pattern '**/*.json': jq package not found` - Install jq with `pip install jq` (optional)
        - `Error loading files with pattern '**/*.ipynb': Permission denied` - Notebook files skipped (normal)
        - `Error loading file *.js` - Some JS files may be skipped (normal)
        
        These warnings don't affect the core functionality. The analyzer will still work with supported file types.
        """)
    
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
    
    if not repo_url:  # Skip if no URL is provided
        return
    
    repo_name = repo_url.split("/")[-1]
    st.info("Cloning the repo...........")
    with tempfile.TemporaryDirectory() as local_path:
        if(clone_git_repo(repo_url,local_path)):
            index, document, file_type_count, file_names = load_and_index_files(local_path)
            
            if(index == None):
                print("No document were found to index.")
                exit()

            print("Repo cloned.....Indexing Files")
            llm = GroqLLM()

            template = '''
            Repo: {repo_name} ({repo_url}) | Conv: {conversation_history} | Docs: {numbered_documents} | Q: {question} | FileCount: {file_type_count} | FileNames: {file_names}

            Instr:
            1. Answer based on context/docs.
            2. Focus on repo/code.
            3. Consider:
                a. Purpose/features - describe.
                b. Functions/code - provide details/samples.
                c. Setup/usage - give instructions.
            4. Unsure? Say "I am not sure".

            Answer:
            '''

            prompt = PromptTemplate(
                template= template,
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

            # Display conversation history
            if hasattr(st.session_state, 'qa_history') and st.session_state.qa_history:
                st.subheader("Conversation History")
                for i, (q, a) in enumerate(st.session_state.qa_history):
                    with st.expander(f"Q{i+1}: {q[:100]}{'...' if len(q) > 100 else ''}", expanded=(i == len(st.session_state.qa_history) - 1)):
                        st.write(f"**Question:** {q}")
                        st.write(f"**Answer:** {a}")
                st.divider()

            if btn or (question_changed and user_question):
                try:
                    with st.spinner("Processing your question..."):
                        user_question = format_questions(user_question)
                        answer = ask_question(user_question, question_context)
                        
                        # Initialize qa_history if it doesn't exist
                        if 'qa_history' not in st.session_state:
                            st.session_state.qa_history = []
                        
                        # Add to conversation history
                        st.session_state.qa_history.append((user_question, answer))
                        st.session_state.conversation_history += f"Question: {user_question}\nAnswer: {answer}\n"
                        st.session_state.conversation_count += 1
                        
                        # Display the new answer immediately
                        st.success("Latest Answer:")
                        st.write(answer)
                        
                        # Force a rerun to update the conversation history display
                        st.rerun()
                        
                except Exception as ex:
                    st.error(f"An error occurred: {ex}")
                    print(f"An error occurred: {ex}")
                    st.session_state.conversation_count += 1  # Increment counter even on error

if __name__ == "__main__":
    main()