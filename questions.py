
from utility import format_document
from repo_reader import search_documents

class QuestionContext:
    def __init__(self,index,documents,llm_chain,model_name,repo_name,repo_url,conversation_history,file_type_count,filenames):
        self.index = index
        self.documents = documents
        self.llm_chain = llm_chain
        self.model_name = model_name
        self.repo_name = repo_name
        self.repo_url = repo_url
        self.conversation_history = conversation_history
        self.file_type_count = file_type_count
        self.filenames = filenames
        
def ask_question(question, context: QuestionContext):
    relevant_docs = search_documents(question, context.index, context.documents, n_results=5)
    numbered_document = format_document(relevant_docs)

    question_context = f"This question is about the github repo '{context.repo_name}' availabe at {context.repo_url}. The most relevant documents are:\n\n{numbered_document}"
    answer_with_sources = context.llm_chain.run(
        model = context.model_name,
        question =  question,
        context = question_context,
        repo_name = context.repo_name,
        repo_url = context.repo_url,
        conversation_history = context.conversation_history,
        numbered_documents = numbered_document,
        file_type_count = context.file_type_count,
        file_names = context.filenames
    )
    return answer_with_sources