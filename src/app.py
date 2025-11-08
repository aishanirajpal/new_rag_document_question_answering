import streamlit as st
from vectorstore import VectorStore
from chatbot import Chatbot
import uuid
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def stream_generator(stream):
    for event in stream:
        if event.event_type == "text-generation":
            yield event.text

def main():
    st.title("Document QA Bot 🤖")
    st.write("Upload PDFs, get summaries, and ask questions!")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vectorstore" not in st.session_state:
        st.session_state["vectorstore"] = None
    if "chatbot" not in st.session_state:
        st.session_state["chatbot"] = None
    if "processed_files" not in st.session_state:
        st.session_state["processed_files"] = []

    # Sidebar
    with st.sidebar:
        st.header("API Keys 🔑")
        cohere_api_key = st.text_input("Cohere API Key", type="password", value=os.getenv("COHERE_API_KEY", ""))
        pinecone_api_key = st.text_input("Pinecone API Key", type="password", value=os.getenv("PINECONE_API_KEY", ""))

        st.header("Your Documents")
        if st.session_state["processed_files"]:
            for file_name in st.session_state["processed_files"]:
                st.write(file_name)
        else:
            st.info("Upload documents to see them here.")

    # Main Panel
    uploaded_files = st.file_uploader(
        "Upload your documents (.pdf, .txt, .docx, .pptx)",
        type=["pdf", "txt", "docx", "pptx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        uploaded_file_names = [f.name for f in uploaded_files]
        files_have_changed = set(uploaded_file_names) != set(st.session_state["processed_files"])

        if files_have_changed:
            if cohere_api_key and pinecone_api_key:
                with st.spinner("Processing PDFs..."):
                    st.session_state["vectorstore"] = VectorStore(uploaded_files, cohere_api_key, pinecone_api_key)
                    st.session_state["chatbot"] = Chatbot(st.session_state["vectorstore"], cohere_api_key)
                    st.session_state["processed_files"] = uploaded_file_names
                    st.session_state["messages"] = [] # Clear history on new upload
                st.success("PDFs processed successfully!")
                st.rerun()
            else:
                st.warning("Please enter your API keys in the sidebar to process the documents.")

    # Display uploaded files and summarize buttons
    if st.session_state["processed_files"]:
        st.subheader("Your Documents")
        for file_name in st.session_state["processed_files"]:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(file_name)
            with col2:
                if st.button("Summarize", key=file_name):
                    if st.session_state.get("chatbot"):
                        with st.spinner(f"Summarizing {file_name}..."):
                            chat_history = st.session_state.get("messages", [])
                            response, _ = st.session_state["chatbot"].summarize(file_name, chat_history)
                            st.session_state["messages"].append({"role": "user", "content": f"Summary of {file_name}", "retrieved_docs": []})
                            st.session_state["messages"].append({"role": "assistant", "content": response, "retrieved_docs": []})
                            st.rerun()
                    else:
                        st.error("Processing not complete or API keys missing.")

    # Display chat history
    for i, message in enumerate(st.session_state["messages"]):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                # Handle streaming response for the last message
                if i == len(st.session_state["messages"]) - 1 and not isinstance(message["content"], str):
                    response_stream = message["content"]
                    full_response = st.write_stream(stream_generator(response_stream))
                    # Store the full response as a string
                    st.session_state["messages"][i]["content"] = full_response
                else:
                    st.markdown(message["content"])

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
                chat_history = st.session_state.get("messages", [])
                response, retrieved_docs = st.session_state["chatbot"].respond(user_query, chat_history)
                st.session_state["messages"].append({"role": "assistant", "content": response, "retrieved_docs": retrieved_docs})
            st.rerun()
        else:
            st.error("Please upload PDFs and provide API keys.")

if __name__ == "__main__":
    main()
