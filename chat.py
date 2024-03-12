import streamlit as st 
from PIL import Image

from langchain import hub
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferWindowMemory

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

# Initializes a conversation chain with a given vector store
def get_conversation_chain(vectorstore):
  memory = ConversationBufferWindowMemory(memory_key='chat_history', return_message=True)
  # rag_prompt_llama = hub.pull("rlm/rag-prompt-llama")
  conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOllama(model=model_select),
    retriever=vectorstore.as_retriever(),
    get_chat_history=lambda h : h,
    # combine_docs_chain_kwargs={"prompt": rag_prompt_llama},
    memory=memory
  )
  return conversation_chain


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
  if st.session_state.conversation == None:
    response = "Please upload documents first to initialize the conversation chain."
  else:
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
      
  # Display AI's response in chat format
  with st.chat_message("assistant"):
    st.write(response)
  # Add AI's response to doc_messages for displaying in UI
  st.session_state['doc_messages'].append({"role": "assistant", "content": response})