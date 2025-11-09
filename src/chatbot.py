import cohere
import uuid
import torch
from httpx import RemoteProtocolError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from transformers import pipeline, AutoTokenizer


class Chatbot:
    def __init__(self, vectorstore, cohere_api_key: str = None):
        self.vectorstore = vectorstore
        self.conversation_id = str(uuid.uuid4())
        self.cohere_api_key = cohere_api_key
        if self.cohere_api_key:
            self.co = cohere.Client(cohere_api_key)
            self.use_hugging_face = False
        else:
            summarizer_model_name = "facebook/bart-large-cnn"
            # Switch to a more powerful generative model
            qa_model_name = "databricks/dolly-v2-3b"
            # Use pipeline's device convention: 0 for first GPU, -1 for CPU
            device = 0 if torch.cuda.is_available() else -1
            self.summarizer = pipeline("summarization", model=summarizer_model_name, device=device)
            self.summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_model_name)
            # Use a text-generation pipeline for this model
            self.qa_pipeline = pipeline("text-generation", model=qa_model_name, device=device, trust_remote_code=True, torch_dtype=torch.bfloat16)
            self.use_hugging_face = True

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RemoteProtocolError)
    )
    def respond(self, user_message: str, chat_history: list = None):
        if self.use_hugging_face:
            retrieved_docs = self.vectorstore.retrieve(user_message)
            context = " ".join([doc['text'] for doc in retrieved_docs])
            
            # Create a proper prompt for the generative model
            prompt = f"""
            Based on the following context, answer the question.
            
            Context:
            {context}
            
            Question:
            {user_message}
            
            Answer:
            """
            
            result = self.qa_pipeline(prompt, max_length=512)
            # The output key for this pipeline is 'generated_text'
            # The model will repeat the prompt in the output, so we need to remove it.
            answer = result[0]['generated_text'].replace(prompt, "")
            return answer, retrieved_docs
        else:
            # Original RAG logic - return the stream directly
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

        if self.use_hugging_face:
            # Bypassing the pipeline's text processing to directly control tokenization.
            # This is a definitive fix for the IndexError.

            # 1. Tokenize the entire document text into token IDs.
            all_token_ids = self.summarizer_tokenizer.encode(document_text, truncation=False)

            # 2. Split the token IDs into chunks safely under the model's 1024 limit.
            max_chunk_length = 1000
            chunks_of_ids = [all_token_ids[i:i + max_chunk_length] for i in range(0, len(all_token_ids), max_chunk_length)]

            summaries = []
            for chunk_of_ids in chunks_of_ids:
                # 3. Create a tensor and move it to the same device as the model.
                input_tensor = torch.tensor([chunk_of_ids]).to(self.summarizer.device)
                
                # 4. Generate summary by calling the model directly.
                summary_ids = self.summarizer.model.generate(
                    input_tensor, 
                    max_length=150, 
                    min_length=30, 
                    num_beams=4,
                    early_stopping=True
                )
                
                # 5. Decode the output token IDs back to a string.
                summary_text = self.summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summaries.append(summary_text)
            
            # 6. Join the summaries from all chunks.
            full_summary = " ".join(summaries)
            return full_summary, []
        else:
            # Use a more direct prompt for summarization - return the stream directly
            message = f"Please provide a concise summary of the following document:\n\n---\n\n{document_text}"
            
            response = self.co.chat_stream(
                message=message,
                model="command-r-plus-08-2024",
                chat_history=chat_history
            )
            return response, []
