import tempfile
from cProfile import label
import PyPDF2
import streamlit as st
import os
import tempfile
import faiss
import torch
import time

from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.summarize.refine_prompts import prompt_template
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import langchain_text_splitters
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from transformers.models.pop2piano.convert_pop2piano_weights_to_hf import model
from transformers.models.siglip.convert_siglip_to_hf import model_name_to_checkpoint

#from projeto1 import user_query

# Em vez de usar dotenv
load_dotenv()

# Acesse as variáveis de ambiente diretamente
#token = os.dotenv("HUGGINGFACEHUB_API_TOKEN")

#if not token:
#    raise ValueError("API_KEY não configurada nas variáveis de ambiente.")

from patsy.util import widen
#from setuptools.command.upload import upload
from xdg.Config import language

#1. Configuração do StreamLit
#1.2 configuração da Página
#st.set_page_config(page_title="Módulo PDF", page_icon="📚", layout="centered", initial_sidebar_state="expanded")
st.title("👨‍🏫 Professor Privado de Contabilidade")
#st.info("💡 Dica: Pergunte sobre conceitos de contabilidade, como 'O que é patrimônio líquido?' ou 'Como funciona a depreciação?'")
#st.button("Botão")
#st.chat_input("Digite sua mensagem")

model_class = "hf_hub"

def model_hf_hub(model = "meta-llama/Meta-Llama-3-8B-Instruct", temperature=0.1):
    llm= HuggingFaceHub(repo_id=model,
                        model_kwargs={
                            "temperature": temperature,
                            "return_full_text": False,
                            "max_new_tokens": 512,
                        })
    return llm


def model_ollama(model = "phi3", temperature = 0.1):
    llm = ChatOllama(model=model, temperature=temperature)
    return llm


# Indexação e recuperação

def config_retriever(uploads):
    # Carregar documetos
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploads:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

    # Divisão em pedaços de texto / split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Embbeding
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    # Armazenamento
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local('vectorstore/db_faiss')

    # Configuração do retrieer
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'fetch_k':4})

    return retriever



# Criação da chain

def config_rag_chain(model_class, retriever):
    ### Carrega a LLM
    if model_class == "hf_hub":
        llm=model_hf_hub()
    elif model_class == "ollama":
        llm=model_ollama()

    # Definição dos Promps
    if model_class.startswith("hf"):
        token_s, token_e = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    else:
        token_s, token_e = "",""

    # consulta -> retriever
    # consulta, histórico do chat -> LLm consulta reformulada -> retriever

    context_q_system_prompt = "Given the following chat history and the follow-up question wich might reference context in the chat history, formulate a standalone question wich can be understood without the chat history. Do NOT answer the question, just reformulate if needed and otherwise return it as is."

    context_q_system_prompt = token_s + context_q_system_prompt
    context_q_user_prompt = "Question: {input}" + token_e
    context_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", context_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", context_q_user_prompt)
        ]
    )

    # Chain para contextualização
    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=retriever, prompt=context_q_prompt
    )


    qa_prompt_template = """Você é um analista senior de licitação e está respondendo ajudando analistas em seus pregões.
    Use os seguintes pedaços de contexto recuperado para responder à pergunta.
    Se você não sabe a resposta, apenas diga que não sabe. Mantenha a resposta concisa.
    Responda em PortuguÇes. \n\n
    Pergunta: {input} \n
    Contexto: {context}"""

    qa_prompt = PromptTemplate.from_template(token_s + qa_prompt_template + token_e)
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    return rag_chain


### Criação do Painel do Módulo de Documentos
uploads = st.sidebar.file_uploader(
    label="Enviar arquivos", type=["pdf"],
          accept_multiple_files=True)



if not uploads:
    st.info("Por Favor, envie algum arquivo para continuar")
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Olá, sou seu professor virtual!")
    ]

if "docs_list" not in st.session_state:
    st.session_state.docs_list = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

start = time.time()
user_query = st.chat_input("Digite sua mensagem aqui...")

if user_query is not None and user_query != "" and uploads is not None:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)
    with st.chat_message("AI"):
        if st.session_state.docs_list != uploads:
            st.session_state.docs_list = uploads
            st.session_state.retriever = config_retriever(uploads)
        rag_chain = config_rag_chain(model_class, st.session_state.retriever)
        result = rag_chain.invoke({"input": user_query, "chat_history": st.session_state.chat_history})
        resp = result['answer']
        st.write(resp)

        sources = result['context']
        for idx, doc in enumerate(sources):
            source = doc.metadata['source']
            file = os.path.basename(source)
            page = doc.metadata.get('page', 'Página não especificada')

            #Fonte1: documento.pdf - p. 2

            ref = f":link: Fonte {idx}: *{file} - p. {page}"
            with st.popover(ref):
                st.caption(doc.page_content)

    st.session_state.chat_history.append(AIMessage(content=resp))

end = time.time()
print ("Tempo: ", end - start)
