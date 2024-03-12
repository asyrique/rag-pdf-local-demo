import streamlit as st
from PyPDF2 import PdfReader
from chat import get_conversation_chain
import tempfile
import os

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import excel

# Extracts and concatenates text from a list of PDF documents
def get_pdf_text(pdf_docs):
  text = ""
  for pdf in pdf_docs:
    try:
      pdf_reader = PdfReader(pdf)
    except (PdfReader.PdfReadError, PyPDF2.utils.PdfReadError) as e:
      print(f"Failed to read {pdf}: {e}")
      continue  # skip to next pdf document in case of read error

    for page in pdf_reader.pages:
      page_text = page.extract_text()
      if page_text:  # checking if page_text is not None or empty string
        text += page_text
      else:
        print(f"Failed to extract text from a page in {pdf}")

  return text

# Splits a given text into smaller chunks based on specified conditions
def get_text_chunks(text):
  text_splitter = RecursiveCharacterTextSplitter(
    separators="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
  )
  chunks = text_splitter.split_text(text)
  return chunks

# Generates embeddings for given text chunks and creates a vector store using FAISS
def get_vectorstore(text_chunks):
  # embeddings = OpenAIEmbeddings()
  embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
  vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
  return vectorstore

# Upload file to Streamlit app for querying
user_uploads = st.file_uploader("Upload your files", accept_multiple_files=True)
docs = []
temp_dir = tempfile.TemporaryDirectory()

if user_uploads is not None:
  if st.button("Upload"):
    with st.spinner("Processing"):

      # Get PDF Text
      raw_text = get_pdf_text(user_uploads)
      # st.write(raw_text)

      for file in user_uploads:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyMuPDFLoader(temp_filepath)
        docs.extend(loader.load())
      # Split documents
      documents = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200).split_documents(docs)
      embedding = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
      vectorstore = FAISS.from_documents(documents=documents, embedding=embedding)

      # Retrieve chunks from text
      # text_chunks = get_text_chunks(raw_text)
      ## st.write(text_chunks)

      # Create FAISS Vector Store of PDF Docs
      # vectorstore = get_vectorstore(text_chunks)

      # Create conversation chain
      st.session_state.conversation = get_conversation_chain(vectorstore)