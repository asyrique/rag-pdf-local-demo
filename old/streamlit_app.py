import streamlit as st 
from PIL import Image
from PyPDF2 import PdfReader

from langchain import hub
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Page configuration for Simple PDF App
st.set_page_config(
  page_title="Simple RAG over PDF",
  page_icon="🧊",
  layout="wide",
  initial_sidebar_state="expanded"
)

st.sidebar.subheader("Setup")
st.sidebar.subheader("Model Selection")
llm_model_options = ['llama2:7b-chat', 'llama2-uncensored','mistral']  # Add more models if available
model_select = st.sidebar.selectbox('Select LLM Model:', llm_model_options, index=0)
st.sidebar.markdown("""\n""")
clear_history = st.sidebar.button("Clear conversation history")

with st.sidebar:
  st.divider()
  st.subheader("Considerations:", anchor=False)
  st.info(
    """
    - Currently only supports PDFs. Include support for .doc, .docx, .csv & .xls files 

    """)

  st.subheader("Updates Required:", anchor=False)
  st.warning("""
    1. Support for multiple PDFs.
    
    2. Use Langchain PDF loader and higher quality vector store for document parsing + reduce inefficient handling.
    
    3. Improve contextual question-answering by developing Prompt Templates - Tendency to hallucinate.

    """
  )

  st.divider()

with st.sidebar:

  st.divider()
  st.write("Made with 🦜️🔗 Langchain and Local LLMs")

if "conversation" not in st.session_state:
  st.session_state.conversation = None

st.markdown(f"""## AI-Assisted Document Analysis 📑 <span style=color:#2E9BF5><font size=5>Beta</font></span>""",unsafe_allow_html=True)
st.write("_A tool built for AI-Powered Research Assistance or Querying Documents for Quick Information Retrieval_")

with st.expander("❔How to use?"):
  st.info("""
  Add files to the RAG system via the sidebar, then you can chat and ask questions about them below.
  """, icon="ℹ️")

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

# Initializes a conversation chain with a given vector store
def get_conversation_chain(vectorstore):
  memory = ConversationBufferWindowMemory(memory_key='chat_history', return_message=True)
  conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOllama(model=model_select),
    retriever=vectorstore.as_retriever(),
    get_chat_history=lambda h : h,
    memory=memory
  )
  return conversation_chain

# Upload file to Streamlit app for querying
user_uploads = st.file_uploader("Upload your files", accept_multiple_files=True)
if user_uploads is not None:
  if st.button("Upload"):
    with st.spinner("Processing"):
      # Get PDF Text
      raw_text = get_pdf_text(user_uploads)
      # st.write(raw_text)

      # Retrieve chunks from text
      text_chunks = get_text_chunks(raw_text)
      ## st.write(text_chunks)  

      # Create FAISS Vector Store of PDF Docs
      vectorstore = get_vectorstore(text_chunks)

      # Create conversation chain
      st.session_state.conversation = get_conversation_chain(vectorstore)

# Initialize chat history in session state for Document Analysis (doc) if not present
if 'doc_messages' not in st.session_state or clear_history:
  # Start with first message from assistant
  st.session_state['doc_messages'] = [{"role": "assistant", "content": "Query your documents"}]
  st.session_state['chat_history'] = []  # Initialize chat_history as an empty list

# Display previous chat messages
for message in st.session_state['doc_messages']:
  with st.chat_message(message['role']):
    st.write(message['content'])

# If user provides input, process it
if user_query := st.chat_input("Enter your query here"):
  # Add user's message to chat history
  st.session_state['doc_messages'].append({"role": "user", "content": user_query})
  with st.chat_message("user"):
    st.markdown(user_query)

  with st.spinner("Generating response..."):
    # Check if the conversation chain is initialized
    if 'conversation' in st.session_state:
      st.session_state['chat_history'] = st.session_state.get('chat_history', []) + [
          {
              "role": "user",
              "content": user_query
          }
      ]
      # Process the user's message using the conversation chain
      result = st.session_state.conversation({
        "question": user_query, 
        "chat_history": st.session_state['chat_history']})
      response = result["answer"]
      # Append the user's question and AI's answer to chat_history
      st.session_state['chat_history'].append({
        "role": "assistant",
        "content": response
      })
    else:
      response = "Please upload a document first to initialize the conversation chain."
    
    # Display AI's response in chat format
    with st.chat_message("assistant"):
      st.write(response)
    # Add AI's response to doc_messages for displaying in UI
    st.session_state['doc_messages'].append({"role": "assistant", "content": response})