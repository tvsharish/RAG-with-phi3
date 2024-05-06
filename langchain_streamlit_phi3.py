from langchain.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import Ollama
import streamlit as st
from PyPDF2 import PdfReader
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
import os


def process_text(text):
  """
    Process the input text by splitting it into chunks, creating embeddings, and building a knowledge base.

    Args:
        text (str): The input text to be processed.

    Returns:
        langchain.vectorstores.faiss.FAISS: A knowledge base from chunks.
  """
  # split the text into chunks using langchain
  text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
  )
  #split text so that model can take it as input for better handling of data
  chunks = text_splitter.split_text(text)
  #initialize GPT4 embeedings
  embeddings = GPT4AllEmbeddings()
  #knowledge base is creasted from FAISS (Facebook AI Similarity Search) which is a library for efficient similarity search and clustering of dense vectors.
  knowledge_base = FAISS.from_texts(chunks, embeddings)
  return knowledge_base

def bot_Response(st,query_text,KB,chain):
    """
    Respond to a user query using a knowledge base and a question-answering (QA) chain model.

    Args:
        st (streamlit): Streamlit object for displaying the response.
        query_text (str): The user's query.
        KB: The knowledge base object for performing similarity search.
        chain: The LLM chain model for generating a response.
    Returns:
        streamlit: The streamlit object with the response displayed.

    This function searches a knowledge base (KB) for relevant content, then generates a response 
    using an LLM chain model and displays it using a streamlit object. The streamlit session state 
    stores the conversation history, including the user's query and the bot's response.
    """
    #Knowledge base will do similarity search for finding most relevant text for given question
    docs = KB.similarity_search(query_text)
    #Question and relevant text passed to chain for response from model
    response = chain.invoke(input={"question": query_text, "input_documents": docs})
    #Add header for answer
    st.subheader("Answer:")
    #Give response to streamlit
    st.write(response["output_text"])
    #Show history of user's question and bot's response to streamlit
    st.session_state.messages.append({"role": "user", "content": query_text})
    st.session_state.messages.append({"role": "assistant", "content": response["output_text"]})
    return st

if __name__=="__main__":
    #give title for the chatbot
    st.title("Chatbot using Phi3")
    #For uploading PDF
    uploaded_file = st.file_uploader('Upload document', type='pdf')
    #For loading phi3 Ollama is used. Make sure to run "ollama run phi3" 
    model_name = "phi3"
    llm = Ollama(model=model_name)
    #Set chat history state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if uploaded_file is not None:
        # All the text from PDF is read by Pypdf
        pdf_reader = PdfReader(uploaded_file)
        
        # store the pdf text in a var
        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text()

        # create a knowledge base object
        KB = process_text(text)
        # Create subheader for question
        st.subheader("Question:")
        # Text box for question
        query_text = st.text_input('Enter your question:', disabled=not uploaded_file)
        count=0
        if query_text:
            #Model loading is done and this chain can be used for multiple questions 
            chain = load_qa_chain(llm, chain_type="stuff")
            #Pass question to bot
            bot_Response(st,query_text,KB,chain)
            count+=1
            #Ask another question button. When clicked text box appears for next question     
            while st.button(f"Ask another question", key=f"button_{str(count)}"):
                try:
                    next_query_text = st.text_input('Enter your next question:', key=f"next_question_{str(count)}", disabled=False)
                    if next_query_text:
                        st.subheader("Question:")
                        st.write(next_query_text)
                        bot_Response(st,next_query_text,KB,chain)
                    count+=1    
                except:
                    count+=1
                    continue

        #Exit button to exit          
        if st.button("Exit"):
            st.stop()
        
