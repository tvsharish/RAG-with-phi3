# Phi3 Chatbot with RAG and Streamlit

This application provides a question-answering chatbot interface where users can upload PDF documents to generate a knowledge base and ask questions directly related to the content of the PDF. 


## Installation

Before you can run the application, you will need to install several dependencies. It is recommended to use a virtual environment.

### Prerequisites

* Python 3.8 or later
* Pip

### Setup

* Clone the repository and navigate to the app directory:
```bash
git clone RAG-with-phi3
cd RAG-with-phi3
```

* Install the required Python packages:
```bash
pip3 install -r requirements.txt
```

* Download and Install ollama in your computer. Installation may vary depending on system:
[Click here to go to Ollama website to download](https://ollama.com/)

### Running the Application

* Run phi3 on one terminal
```bash
ollama run phi3
```

* To run the app, use the following command in another terminal:
```bash
python3 -m streamlit run langchain_streamlit_phi3.py
```

### Usage
* Start the Application: Open your terminal and run the app with Streamlit.
* Upload a PDF: Use the 'Upload document' button to upload a PDF file from which you want to extract information.
* Ask Questions: After uploading the document, enter your questions in the text input field. The chatbot will process your questions and return answers based on the PDF content.
* Interactive Chat: The application maintains a history of your questions and the chatbot's responses. You can continue the conversation by asking more questions.
* Exit: You can terminate the session using the 'Exit' button in the application.
