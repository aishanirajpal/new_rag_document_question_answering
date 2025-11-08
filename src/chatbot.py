import cohere
import uuid
from httpx import RemoteProtocolError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class Chatbot:
    def __init__(self, vectorstore, cohere_api_key: str):
        self.vectorstore = vectorstore
        self.conversation_id = str(uuid.uuid4())
        self.co = cohere.Client(cohere_api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RemoteProtocolError)
    )
    def respond(self, user_message: str, chat_history: list = None):
        # Check for summarization intent
        is_summarization_request = any(word in user_message.lower() for word in ["summary", "summarize"])

        if is_summarization_request:
            # For summarization, get the full text of all documents
            full_texts = [doc['text'] for doc in self.vectorstore.doc_texts]
            source_files = [doc['source'] for doc in self.vectorstore.doc_texts]
            
            # Create a single context string with clear document separation
            combined_text = ""
            for i, text in enumerate(full_texts):
                combined_text += f"--- Document: {source_files[i]} ---\n\n{text}\n\n"
            
            message = f"Please provide a concise summary of the following documents:\n\n{combined_text}"
            
            response = self.co.chat_stream(
                message=message,
                model="command-r-plus-08-2024",
                chat_history=chat_history
            )
            return response, []
        
        # Original RAG logic for other questions
        retrieved_docs = self.vectorstore.retrieve(user_message)

        # Augment the prompt with the list of available documents
        source_files = list(set(doc['source'] for doc in self.vectorstore.doc_texts))
        files_prompt = f"The user has uploaded the following documents: {', '.join(source_files)}. Please use these filenames when referencing documents in your answer."
        augmented_message = f"{files_prompt}\n\nQuestion: {user_message}"

        response = self.co.chat_stream(
            message=augmented_message,
            model="command-r-plus-08-2024",
            documents=retrieved_docs,
            chat_history=chat_history
        )
        return response, retrieved_docs

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RemoteProtocolError)
    )
    def summarize(self, file_name: str, chat_history: list = None):
        document_text = self.vectorstore.get_document_text(file_name)
        if not document_text:
            return "Document not found.", []

        # Use a more direct prompt for summarization
        message = f"Please provide a concise summary of the following document:\n\n---\n\n{document_text}"
        
        response = self.co.chat_stream(
            message=message,
            model="command-r-plus-08-2024",
            chat_history=chat_history
        )
        return response, []
