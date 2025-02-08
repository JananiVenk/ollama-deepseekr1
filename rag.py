import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

prompt_template="""
You are an expert research assistant. Use the provided context to answer the query.
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query:{user_query}
Document context:{document_context}
Answer:"""

PDF_STORAGE_PATH="document_store/pdfs/"
EMBEDDING_MODEL=OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB=InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL=OllamaLLM(model="deepseek-r1:1.5b")

def save_uploaded_file(uploaded_file):
    file_path=PDF_STORAGE_PATH+uploaded_file.name
    with open(file_path,"wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_document(file_path):
    document_loader=PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query):
    return  DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query,context_documents):
    context_text="\n\n".join([doc.page_content for doc in context_documents])
    conversation_template=ChatPromptTemplate.from_template(prompt_template)
    response_chain=conversation_template|LANGUAGE_MODEL
    return response_chain.invoke({"user_query":user_query,"document_context":context_text})

st.title("AI Document Processor")
st.markdown("### Intelligent Document Assistant")
st.markdown("----")

uploaded_pdf=st.file_uploader(
    "Upload Research Document",
    type="pdf",
    help="Select a pdf document for analysis",
    accept_multiple_files=False
)

if uploaded_pdf:
    saved_path=save_uploaded_file(uploaded_pdf)
    raw_documents=load_pdf_document(saved_path)
    processed_chunks=chunk_documents(raw_documents)
    index_documents(processed_chunks)

    st.success("Document processed successfully! Ask your questions below...")

    user_input=st.chat_input("Enter your question about the document")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
    
        with st.spinner("Analyzing documents..."):
            relevant_docs=find_related_documents(user_input)
            ai_response=generate_answer(user_input,relevant_docs)
        
        with st.chat_message("assistant"):
            st.write(ai_response)
            
    
    