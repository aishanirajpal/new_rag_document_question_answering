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
        retrieved_docs = self.vectorstore.retrieve(user_message)
        response = self.co.chat_stream(
            message=user_message,
            model="command-r-plus-08-2024",
            documents=retrieved_docs,
            conversation_id=self.conversation_id,
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

        message = f"Summarize the following document in 400-500 words: {document_text}"
        response = self.co.chat_stream(
            message=message,
            model="command-r-plus-08-2024",
            conversation_id=self.conversation_id,
            chat_history=chat_history
        )
        return response, []
