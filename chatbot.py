import json
import logging
import time
from io import BytesIO

import numpy as np
import requests
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sklearn.metrics.pairwise import cosine_similarity

# Logging configuration
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Configuration
gpt_server = "http://127.0.0.1:11434"
gpt_url = f"{gpt_server}/v1/chat/completions"


def read_documentation(uploaded_files):
    docs_content = ''
    for uploaded_file in uploaded_files:
        logging.info(f"Parsing file : {uploaded_file.name}")
        if uploaded_file.type == "application/pdf":
            # Extract text from PDF using PyPDF2
            pdf_reader = PdfReader(BytesIO(uploaded_file.read()))
            for page in pdf_reader.pages:
                docs_content += page.extract_text()
        else:
            docs_content += uploaded_file.getvalue().decode("utf-8") + '\n'
    return docs_content


def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(separators="\n", chunk_size=1000, chunk_overlap=150, length_function=len)
    return text_splitter.split_text(text)


def generate_embeddings(text):
    embeddings = st.session_state.embedding_model.get_text_embedding(text)
    return np.array(embeddings)


def get_relevant_context(query, doc_embeddings, doc_chunks):
    query_embedding = generate_embeddings(query).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, doc_embeddings)
    relevant_indices = similarities.argsort()[0][-3:][::-1]  # Get top 3 most relevant chunks
    relevant_context = "\n\n".join([doc_chunks[i] for i in relevant_indices])
    return relevant_context


def ask_gpt(message, relevant_context, chat_history):
    # Add chat history to the payload
    payload = {
        "model": st.session_state.gpt_model,
        "temperature": float(st.session_state.temperature),
        "max_tokens": int(st.session_state.gpt_response_length),
        "messages": [
            {"content": st.session_state.prompt_input, "role": "system"},
            {"content": chat_history, "role": "system"},  # Pass history to provide chat context
            {"content": message + "\n\nContext: " + relevant_context, "role": "user"},
        ]
    }
    logging.info(f"Asking LLM about: {message}")

    max_retries = 5
    retry_count = 0
    msg_response = None

    while retry_count < max_retries:
        try:
            response = requests.post(gpt_url, json=payload, headers={'Content-Type': 'application/json'})
            logging.info(f"Response from LLM: {response.status_code}")
            msg_response = json.loads(response.text)['choices'][0]['message']['content']
            break
        except Exception as e:
            log.error(f"Exception in making HTTP call: {str(e)}")
            retry_count += 1
            time.sleep(2)
    if msg_response is None:
        logging.info("Unable to get response from GPT")
        msg_response = "Unable to get response from GPT"
    return msg_response


def main():
    st.title("Documentation Q&A Bot")
    st.write("Upload your documentation files in the sidebar and ask questions.")

    # Sidebar for file upload
    with st.sidebar:
        if "documentation_loaded" not in st.session_state:
            st.session_state.documentation_loaded = False

        if not st.session_state.documentation_loaded:
            uploaded_files = st.file_uploader("Upload Documentation Files", accept_multiple_files=True)
            if uploaded_files:
                st.session_state.embedding_model = HuggingFaceEmbedding(
                    model_name="sentence-transformers/all-MiniLM-L6-v2")
                documentation = read_documentation(uploaded_files)
                chunks = split_text(documentation)
                embeddings = np.array([generate_embeddings(chunk) for chunk in chunks])

                # Save in session state
                st.session_state.documentation = documentation
                st.session_state.chunks = chunks
                st.session_state.embeddings = embeddings
                st.session_state.documentation_loaded = True
                st.success("Documentation loaded successfully. You can now ask questions.")
        else:
            st.write("Documentation is loaded. You can now ask questions.")

            # Show the documentation content in the sidebar
            with st.expander("Uploaded Documentation"):
                st.text_area("Documentation Content", st.session_state.documentation, height=300, disabled=True)

        # LLM Model selection
        st.session_state.gpt_model = st.selectbox("Select LLM Model", ["mistral-openorca", "llama3.1", "codellama", "deepseek-coder-v2"])

        # Response length
        st.session_state.gpt_response_length = st.text_input("Response Length", 1000)

        # Temperature
        st.session_state.temperature = st.text_input("Temperature", 0)

        # Prompt input text area
        st.session_state.prompt_input = st.text_area("Enter additional prompt:",
                                                     "You are a helpful assistant who is an expert in understanding documents and providing answers based on those documents.\n"
                                                     "Understand the following document and respond to answers based on this context only.\n"
                                                     "If you can't find the answer in this context, just say 'I don't know'.\n",
                                                     height=300)

    if st.session_state.documentation_loaded:
        query = st.text_input("Enter your question:")
        if query:
            relevant_context = get_relevant_context(query, st.session_state.embeddings, st.session_state.chunks)
            if "history" not in st.session_state:
                st.session_state.history = ""
            if "response_history" not in st.session_state:
                st.session_state.response_history = ""
            answer = ask_gpt(query, relevant_context, st.session_state.history)
            response = (f"**Question**: {query}, Model: {st.session_state.gpt_model}, "
                        f"Temperature: {st.session_state.temperature}, "
                        f"At: {time.ctime()}\n\n**Answer**: {answer}")
            # Update history
            st.session_state.history += answer + "\n\n---\n\n"
            st.session_state.response_history = response + "\n\n---\n\n" + st.session_state.response_history
            st.write(st.session_state.response_history)


if __name__ == '__main__':
    main()
