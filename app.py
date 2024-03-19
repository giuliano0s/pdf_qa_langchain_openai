import streamlit as st

# Model/parser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Embedding/vector store
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader

import time

# System
import os

#%%#################################################### FUNCTIONS

def check_last_api_input():
    lst = [x for x in st.session_state.log_messages if x in ['API Validated','Please, insert a valid API key']]
    return lst

def write_logs(new_message=False):
    if new_message:
        st.session_state.log_messages.insert(0, new_message)

    for message in st.session_state.log_messages:
        st.sidebar.text(message)

def validate_key():
    try:
        ChatOpenAI(openai_api_key=OPENAI_API_KEY).invoke("Testing key")
        return True
    except:
        return False

#%%#################################################### SIDEBAR

# Title
st.sidebar.title('PDF Q&A')

# Start log messages
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = ['Please, insert a valid API key']

# Reload files button
reload_button = st.sidebar.button('Reload files')
if reload_button:
    try:
        st.session_state.pop('vector')
        st.session_state.log_messages = ['Please, validate your API key again']
        st.sidebar.write('Files updated')
    except:
        st.sidebar.write('Nothing to update')

# Validade API key button
OPENAI_API_KEY = st.sidebar.text_input('OPENAI_API_KEY:')

validate_button = st.sidebar.button('Validate key')
if validate_button:
    st.session_state.valid_key = validate_key()

    if st.session_state.valid_key:
        st.session_state.log_messages.insert(0, 'API Validated')
    else:
        st.session_state.log_messages.insert(0, 'Please, insert a valid API key')

else:
    if 'valid_key' not in st.session_state:
        st.session_state.valid_key = False

    st.sidebar.markdown('### Logs')
    write_logs()

#%%#################################################### READ FILES

if ('vector' not in st.session_state) and (st.session_state.valid_key):

    # Scan folder
    files = os.listdir('./data/unprocessed_data/')
    files = [x for x in files if x[-4:]=='.pdf']
    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)

    if len(files)>0:
        write_logs(f'{len(files)} new files found, processing and merging')

        #Load files
        loaders = [PyPDFLoader('./data/unprocessed_data/' + x) for x in files]

        docs = []
        for loader in loaders:
            docs.extend(loader.load())

        #Process
        llm = ChatOpenAI(openai_api_key = OPENAI_API_KEY, model_name = 'gpt-3.5-turbo-16k', temperature=0.5)
        splits_prompt = ChatPromptTemplate.from_messages([
            ("system", "Formate o texto para que fique entendível para usar de contexto em uma LLM. me dê apenas o output como resposta:"),
            ("user", "{input}")
        ])
        splits_parser = StrOutputParser()
        splits_chain = splits_prompt | llm | splits_parser
        for n in range(len(docs)):
            docs[n].page_content = splits_chain.invoke({"input": f"{docs[n].page_content}"})

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)

        #Vector merging
        vector = FAISS.from_documents(splits, embeddings)
        old_vector = FAISS.load_local("./data/faiss_index", embeddings, allow_dangerous_deserialization=True)
        vector.merge_from(old_vector)

        vector.save_local("./data/faiss_index")

        for file in files:
            os.replace(f"./data/unprocessed_data/{file}", f"./data/processed_data/{file}")

        write_logs('Finished processing and merging')

    else:
        vector = FAISS.load_local("./data/faiss_index", embeddings, allow_dangerous_deserialization=True)
        time.sleep(1)
        write_logs('0 new files found, loading old vector')

    st.session_state.vector = vector

#%%#################################################### MODEL

if ('vector' in st.session_state) and (st.session_state.valid_key) and ('started' not in st.session_state):
    #Chain creation

    st.session_state.started = True

    llm = ChatOpenAI(openai_api_key = OPENAI_API_KEY, model_name = 'gpt-3.5-turbo-16k', temperature=0.5)

    prompt = ChatPromptTemplate.from_template(
        """"Você deve memorizar os editais do IFRS canoas para responder questões sobre ele.

            Cursos de Licenciatura, Bacharel e Tecnólogo são considerados cursos de nível superior.

            Dê informações completas.

            <context>
            {context}
            </context>

            Historico de mensagens: {history}

            Pergunta: {input}""")

    #retrieval
    document_chain = create_stuff_documents_chain(llm, prompt)

    retriever = st.session_state.vector.as_retriever(search_kwargs={'k': 10, 'score_treshold': 0.9},
                                                        search_type="similarity")
    st.session_state.retrieval_chain = create_retrieval_chain(retriever, document_chain)

#%%#################################################### CHAT

if ('vector' in st.session_state) and (st.session_state.valid_key) and ('started' in st.session_state):

    # Init. chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_hist = []
        st.markdown('<div class="chat-messages">' + '<br>'.join(reversed(st.session_state.messages)) + '</div>', unsafe_allow_html=True)

    # Messages form
    with st.form("form_message", clear_on_submit=True):

        st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
        user_message = st.text_input('Type your message:', key="message_input")
        send_message = st.form_submit_button('Send message')
        st.markdown('</div>', unsafe_allow_html=True)

    if send_message and user_message:
            st.session_state.messages.append(f"User: {user_message}")
            st.session_state.chat_hist.append({'user': user_message})
            st.session_state.messages.append(f"_______________________________________________________")

            response = st.session_state.retrieval_chain.invoke({"input": user_message, "history": st.session_state.chat_hist})
            st.session_state.messages.append(f"ChatGPT: {response['answer']}")
            st.session_state.chat_hist.append({'agent': response['answer']})
            st.session_state.messages.append(f"_______________________________________________________")



    # Show messages
    st.markdown('<div class="chat-messages">' + '<br><br>'.join(reversed(st.session_state.messages)) + '</div>', unsafe_allow_html=True)