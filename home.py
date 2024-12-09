import streamlit as st
import os

# Configuração do título
st.set_page_config(page_title="Home", layout="centered")
import os

script_path = "pages/"

# Página inicial
st.title("Bem-vindo ao Hub de Projetos")
st.markdown("Escolha um dos projetos abaixo para começar:")

# Menu de navegação
project_options = {
    "Projeto 1": "pages/project1.py",
    "Projeto 2": "pages/project2.py",
}

# Seleção do projeto
project_selected = st.selectbox("Selecione um projeto:", ["Selecione"] + list(project_options.keys()))

if project_selected != "Selecione":
    script_path = project_options[project_selected]
    with open(script_path) as f:
        exec(f.read())  # Executa o script do projeto selecionado
