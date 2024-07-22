import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from decouple import config
from langchain.memory import ConversationBufferWindowMemory

def create_prompt(question, context):
    template="""You are a friendly asistant expert in building conversational assistants using generative ai. 
        Use the context provided for reference and answer to questions.
        Context: {context}
        
        Human: {question}
        AI:"""
    
    template = template.format(context=context, question=question)
    
    prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=template
        
    )

def create_question_prompt(question):
    template="""
        Human: {question}
        AI:"""
    
    template = template.format(question=question)
    
    prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=template
        
    )


# prompt = PromptTemplate(
#         input_variables=["chat_history", "question", "context"],
#         template="""You are a friendly asistant expert in building conversational assistants using generative ai. 
#         Answer to questions in precise manner.
#         Context: {context}
        
#         chat_history: {chat_history},
#         Human: {question}
#         AI:"""
#     )


    return prompt

def get_context():
    import re
    file = open('context1.txt', 'r')
    content = file.read()

    content = re.sub("PG&E", "XYZ", content)
    content = re.sub("PGE", "XYZ", content)

    return content

def call_llm(question, use_context):
    llm = ChatOpenAI(openai_api_key=config("OPENAI_API_KEY"), model="gpt-4-turbo")

    memory = ConversationBufferWindowMemory(memory_key="chat_history", k=4)
    if use_context:
        prompt_to_use = create_prompt(get_context(), question)
    else:
        prompt_to_use = create_question_prompt(question)

    llm_chain = LLMChain(
        llm=llm,
        memory=memory,
        prompt=prompt_to_use
    )
    return llm_chain.predict(question=question)


st.set_page_config(
    page_title="My BOT",
    page_icon="🤖",
    layout="wide"
)

st.title("My BOT")

use_context = st.checkbox("Use Context file?")

# check for messages in session and create if not exists
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello there, I am your BOT"}
    ]

# Display all messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_prompt = st.chat_input()

if user_prompt is not None:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            ai_response = call_llm(user_prompt, use_context)
            st.write(ai_response)
    new_ai_message = {"role": "assistant", "content": ai_response}
    st.session_state.messages.append(new_ai_message)

