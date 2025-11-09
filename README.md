# RAG-Based Document Question Answering System 🤖📄

This project implements a flexible **Retrieval-Augmented Generation (RAG) chatbot** with a dual-backend system, all wrapped in a professional and modular Streamlit user interface.

It allows users to upload documents, ask questions based on their content, and receive accurate, context-aware answers.

---

## Key Features ✨

- **Dual-Backend System**:
    - **Cloud-Based**: Uses **Cohere** for powerful language processing and embeddings, and **Pinecone** for scalable, managed vector storage.
    - **Fully Local**: Uses open-source **Hugging Face** models (`all-MiniLM-L6-v2` for embeddings, `databricks/dolly-v2-3b` for generation) and **FAISS** for local, in-memory vector search.
- **Dynamic Backend Switching**: Automatically uses the local Hugging Face backend if Cohere/Pinecone API keys are not provided.
- **Multi-Format Document Support**: Extracts text from `PDF`, `DOCX`, `PPTX`, and `TXT` files.
- **Professional UI**: A clean, modern, and modular interface built with Streamlit, featuring:
    - A tabbed layout to separate "Question & Answer" from "Document Summarization".
    - A "Clear Chat History" button.
    - A responsive, professional design with custom CSS.

---

## How to Use 🚀

You can run this application locally on your machine or deploy it to the web.

### 1. Offline (Local Setup)

Follow these steps to run the project on your own computer.

#### a. Clone the Repository
Clone the repository to your local system:
```bash
git clone https://github.com/aishanirajpal/new_rag_document_question_answering
cd new_rag_document_question_answering
```

#### b. Create and Activate a Virtual Environment 🏗️
Create a virtual environment to isolate project dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### c. Install Dependencies 📦
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

#### d. Run the Application 🏃‍♂️
Launch the Streamlit application from the project's root directory:
```bash
streamlit run src/app.py
```

#### e. Access the Application 🌐
Open your browser and navigate to the local URL provided by Streamlit, typically `http://localhost:8501`.

#### f. Choose Your Backend
- **For the local Hugging Face experience**, simply leave the API key fields in the sidebar blank. The first time you run the app, it will download several gigabytes of open-source models. This is a one-time process.
- **For the cloud-based Cohere/Pinecone experience**, enter your API keys in the sidebar.

---

### 2. Online (Streamlit Cloud Deployment)

You can deploy this application to Streamlit's free Community Cloud. **Note: Only the Cohere/Pinecone mode is suitable for the free tier**, as the local Hugging Face models are too large for the provided resources.

#### a. Push to GitHub
Ensure your project is pushed to a public GitHub repository.

#### b. Deploy on Streamlit Cloud
1.  Sign up for [Streamlit Community Cloud](https://streamlit.io/cloud).
2.  Click "New app" and connect to your GitHub repository.
3.  Select the correct branch and set the main file path to `src/app.py`.
4.  Under "Advanced settings," add your `COHERE_API_KEY` and `PINECONE_API_KEY` to the secrets management.
5.  Click "Deploy!".

#### c. Access Your Deployed App
Once deployed, you can access your application via its public Streamlit Cloud URL:

**[https://documents-q-a-chatbot.streamlit.app/]**

---

# Project Structure 📁
```bash
.
├── src/
│   ├── app.py                # Main Streamlit UI application
│   ├── vectorstore.py        # Handles document processing, embedding, and retrieval
│   ├── chatbot.py            # Handles response generation for both backends
│   └── style.css             # Custom CSS for professional UI styling
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation
└── .gitignore
```

### Future Enhancements 🚧
- Add support for multi-language documents.
- Enhance the UI with multi-document support and export options for chat history.
- Integrate additional vector databases for broader compatibility.

## Contributing 🤝

🚀 We warmly welcome contributions to enhance this project! Whether it's fixing bugs, adding new features, or improving documentation, your efforts will help make this project better for everyone. Let's collaborate and build something amazing together! 🌟✨

### License  📜

This project is licensed under the Apache License.

### Acknowledgments 🙏

- **Cohere AI** & **Pinecone** for their powerful, cloud-based AI infrastructure. 🧠⚡
- **Hugging Face** for providing the open-source models and libraries that power the local backend. 🤗
- **LangChain** & **FAISS** for the core text splitting and vector search capabilities. 🔗
- **Streamlit** for making it easy to build beautiful, interactive data apps. 📊🎉