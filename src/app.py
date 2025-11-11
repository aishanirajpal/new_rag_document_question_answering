import streamlit as st
from vectorstore import VectorStore
from chatbot import Chatbot
import uuid
import os
from dotenv import load_dotenv
import types

# Load environment variables from .env file
load_dotenv()

# Function to load and inject CSS
def add_css(file_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, file_name)
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Generator for streaming responses
def stream_generator(stream):
    for event in stream:
        if event.event_type == "text-generation":
            yield event.text

@st.cache_resource
def get_vectorstore(uploaded_files, cohere_api_key, pinecone_api_key):
    # Convert tuple of files to list
    return VectorStore(list(uploaded_files), cohere_api_key, pinecone_api_key)

@st.cache_resource
def get_chatbot(_vectorstore, cohere_api_key):
    return Chatbot(_vectorstore, cohere_api_key)

def main():
    st.set_page_config(
        page_title="RAG Q/A ChatBot",
        page_icon="📄",
        layout="wide"
    )
    add_css("style.css")

    st.title("RAG Q/A ChatBot")
    st.markdown("<h6 style='text-align: center; font-weight: 600;'>Upload documents, get summaries, and ask questions!</h6>", unsafe_allow_html=True)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vectorstore" not in st.session_state:
        st.session_state["vectorstore"] = None
    if "chatbot" not in st.session_state:
        st.session_state["chatbot"] = None
    if "processed_files" not in st.session_state:
        st.session_state["processed_files"] = []
    if "summary_results" not in st.session_state:
        st.session_state["summary_results"] = {}

    # Sidebar
    with st.sidebar:
        st.header("API Keys")
        cohere_api_key = st.text_input("Cohere API Key", type="password", value=os.getenv("COHERE_API_KEY", ""))
        pinecone_api_key = st.text_input("Pinecone API Key", type="password", value=os.getenv("PINECONE_API_KEY", ""))

        st.header("Your Documents")
        if st.session_state["processed_files"]:
            for file_name in st.session_state["processed_files"]:
                st.write(file_name)
        else:
            st.info("Upload documents to see them here.")

        # Use markdown to center the button
        st.markdown("""
            <style>
                div[data-testid="stSidebar"] .stButton {
                    display: flex;
                    justify-content: center;
                }
            </style>
        """, unsafe_allow_html=True)
        
        if st.button("Clear Chat History"):
            st.session_state["messages"] = []
            st.rerun()

    # Main Panel
    uploaded_files = st.file_uploader(
        "Upload your documents (.pdf, .txt, .docx, .pptx)",
        type=["pdf", "txt", "docx", "pptx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        # Convert list of files to tuple for caching
        uploaded_files_tuple = tuple(uploaded_files)
        uploaded_file_names = [f.name for f in uploaded_files]
        files_have_changed = set(uploaded_file_names) != set(st.session_state.get("processed_files", []))

        if files_have_changed:
            with st.spinner("Processing documents..."):
                st.session_state["vectorstore"] = get_vectorstore(uploaded_files_tuple, cohere_api_key, pinecone_api_key)
                st.session_state["chatbot"] = get_chatbot(st.session_state["vectorstore"], cohere_api_key if cohere_api_key else None)
                st.session_state["processed_files"] = uploaded_file_names
                st.session_state["messages"] = [] # Clear history on new upload
                st.session_state["summary_results"] = {} # Clear summaries on new upload
            st.success("Documents processed successfully!")
            st.rerun()

    # Create tabs
    qa_tab, summary_tab = st.tabs(["Question & Answer", "Document Summarization"])

    with qa_tab:
        st.subheader("Ask a Question")
        # Display chat history
        for i, message in enumerate(st.session_state["messages"]):
            with st.chat_message(message["role"]):
                # Check if the content is a stream (generator)
                if isinstance(message["content"], types.GeneratorType):
                    full_response = st.write_stream(stream_generator(message["content"]))
                    # Store the full response as a string once it's complete
                    st.session_state["messages"][i]["content"] = full_response
                else:
                    st.markdown(message["content"])

                if message["role"] == "assistant":
                    retrieved_docs = message.get("retrieved_docs", [])
                    if retrieved_docs:
                        with st.expander("Retrieved Documents"):
                            for doc in retrieved_docs:
                                st.info(f"Source: {doc['source']}\n\nText: {doc['text']}")

        # Chat input
        if user_query := st.chat_input("Ask a question based on the documents..."):
            if st.session_state.get("chatbot"):
                st.session_state["messages"].append({"role": "user", "content": user_query, "retrieved_docs": []})
                with st.spinner("Generating response..."):
                    # Format chat history for Cohere API
                    cohere_history = []
                    for msg in st.session_state["messages"]:
                        if msg["role"] == "user":
                            cohere_history.append({"role": "USER", "message": msg["content"]})
                        elif msg["role"] == "assistant":
                            # Ensure content is a string before appending
                            if isinstance(msg["content"], str):
                                cohere_history.append({"role": "CHATBOT", "message": msg["content"]})

                    response, retrieved_docs = st.session_state["chatbot"].respond(user_query, cohere_history)
                    st.session_state["messages"].append({"role": "assistant", "content": response, "retrieved_docs": retrieved_docs})
                st.rerun()
            else:
                st.error("Please upload documents first.")

    with summary_tab:
        st.subheader("Get a Summary")
        if st.session_state["processed_files"]:
            for file_name in st.session_state["processed_files"]:
                if st.button(f"Summarize {file_name}", key=file_name):
                    if st.session_state.get("chatbot"):
                        with st.spinner(f"Summarizing {file_name}..."):
                            # Format chat history for Cohere API
                            cohere_history = []
                            for msg in st.session_state["messages"]:
                                if msg["role"] == "user":
                                    cohere_history.append({"role": "USER", "message": msg["content"]})
                                elif msg["role"] == "assistant":
                                    cohere_history.append({"role": "CHATBOT", "message": msg["content"]})
                            
                            response, _ = st.session_state["chatbot"].summarize(file_name, cohere_history)
                            # Store the summary result in its own state, not in the chat history
                            st.session_state["summary_results"][file_name] = response
                            st.rerun() # Rerun to display the summary immediately below the button
                    else:
                        st.error("Processing not complete or an error occurred.")

                # Display the summary if it exists for this file
                if file_name in st.session_state["summary_results"]:
                    summary_content = st.session_state["summary_results"][file_name]
                    with st.expander(f"Summary for {file_name}", expanded=True):
                        # Check if the summary content is a stream
                        if isinstance(summary_content, types.GeneratorType):
                            full_summary = st.write_stream(stream_generator(summary_content))
                            # Store the full summary as a string once complete
                            st.session_state["summary_results"][file_name] = full_summary
                        else:
                            st.markdown(summary_content)

        else:
            st.info("Upload documents to see summarization options.")


if __name__ == "__main__":
    main()
