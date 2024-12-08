import streamlit as st
from langchain.chains.summarize.refine_prompts import prompt_template
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import torch
from langchain_huggingface import ChatHuggingFace
from langchain_community.llms import HuggingFaceHub

import os

# Em vez de usar dotenv
# load_dotenv()

# Acesse as variáveis de ambiente diretamente
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not token:
    raise ValueError("API_KEY não configurada nas variáveis de ambiente.")

from patsy.util import widen
#from setuptools.command.upload import upload
from xdg.Config import language

#1. Configuração do StreamLit
#1.2 configuração da Página
st.set_page_config(page_title="Assistente de Estudantes de Contabilidade", page_icon="📚", layout="centered", initial_sidebar_state="expanded")
st.title("👨‍🏫 Professor Privado de Contabilidade")
st.info("💡 Dica: Pergunte sobre conceitos de contabilidade, como 'O que é patrimônio líquido?' ou 'Como funciona a depreciação?'")
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


def model_response(user_query, chat_history, model_class):
    #Carregamento da LLM
    if model_class == "hf_hub":
        llm= model_hf_hub()
    elif model_class =="openai":
        llm=model_openai()
    elif model_class=="ollama":
        llm=model_ollama()

    # Definição dos prompts
    language="Português Brasileiro"
    system_prompt=f"Você é um professor de Contabilidade, especialista em Excel e Matemática. Você está ajudando estudantes com suas atividades da faculdade. Responda em {language}. Escreva informalmente."

    if model_class.startswith("hf"):
        user_prompt = f"<begin_of_text|><start_header_id|>user<|end_header_id|>\n{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    else:
        user_prompt = "{input}"

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", user_prompt)
    ])

    chain = prompt_template | llm | StrOutputParser()

    return chain.stream({
        "chat_history": chat_history,
        "input": user_query,
        "language": language
    })


# Verificando se chat_history está no session_state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Como posso te ajudar nos seus estudos hoje?")
    ]

# Exibindo as mensagens do chat
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI", avatar="🧑‍🏫"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human", avatar="🧑‍🎓"):
            st.write(message.content)

# Entrada do usuário
user_query = st.chat_input("Fale com seu professor aqui")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human", avatar="🧑‍🎓"):
        st.markdown(user_query)
    # Mostra feedback visual enquanto processa abaixo da pergunta
    processing_placeholder = st.empty()  # Cria um espaço reservado para a mensagem de processamento
    processing_placeholder.write("🔍 Processando sua dúvida...")
    with st.chat_message("AI", avatar="🧑‍🏫"):
        resp = model_response(user_query, st.session_state.chat_history, model_class)
        for response_part in resp:
            st.markdown(response_part)  # Stream da resposta
    st.session_state.chat_history.append(AIMessage(content=response_part))
    processing_placeholder.empty()

# Indexação e Recuperação




# Criação de painel lateral na interface
