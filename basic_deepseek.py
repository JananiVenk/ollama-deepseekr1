import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)

st.title("DeepSeek Companion")
st.caption("AI programmer with debugging skills")

with st.sidebar:
    st.markdown("### Model Capablities")
    st.markdown("""
                    - Python Expert
                    - Debugging Assistant
                    - Code Documentation
                    - Solution Design""")
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.com/) | [Langchain](https://python.langchain.com/docs/introduction/)")

llm=ChatOllama(model="deepseek-r1:1.5b",
               base_url="http://localhost:11434",
               temperature=0.3)

system_message=SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, correct solutions."
    "with strategic print statements for debugging. Always respond in english"
)

if "message_log" not in st.session_state:
    st.session_state.message_log=[{'role':'ai','content':"Hi! I'm Deepseek.How can I help you code?"}]

chat_container=st.container()

with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

user_input=st.chat_input("Type your coding question...")

def generate_ai_response(prompt_chain):
    processing_pipeline=prompt_chain|llm|StrOutputParser()
    return processing_pipeline.invoke({})

def built_prompt_chain():
    prompt_sequence=[system_message]
    for message in st.session_state.message_log:
        if message['role']=='user':
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(message['content']))
        elif message['role']=='ai':
            prompt_sequence.append(AIMessagePromptTemplate.from_template(message['content']))
    return ChatPromptTemplate.from_messages(prompt_sequence)

if user_input:
    st.session_state.message_log.append({'role':'user','content':user_input})

    with st.spinner("Processing..."):
        prompt_chain=built_prompt_chain()
        ai_response=generate_ai_response(prompt_chain)
    
    st.session_state.message_log.append({'role':'ai','content':ai_response})

    st.rerun()