import streamlit as st
import os

# Configuração do título
st.set_page_config(page_title="Home", layout="centered")

# Diretório onde os projetos estão localizados
scripts_dir = "pages/"

# Página inicial
st.title("Bem-vindo ao Hub de Projetos")
st.markdown("Escolha um dos projetos abaixo para começar:")

# Menu de navegação
project_options = {
    "Projeto 1": "projeto1.py",  # Caminho do arquivo
    "Projeto 2": "projeto2.py",
}

# Seleção do projeto
project_selected = st.selectbox("Selecione um projeto:", ["Selecione"] + list(project_options.keys()))

if project_selected != "Selecione":
    script_name = project_options[project_selected]
    script_path = os.path.join(scripts_dir, script_name)
    
    # Verificando se o arquivo existe
    if os.path.exists(script_path):
        # Carregando o conteúdo do script e exibindo no Streamlit
        with open(script_path, "r") as file:
            code = file.read()
            st.code(code, language="python")  # Exibe o conteúdo do script no Streamlit
    else:
        st.error(f"Erro: O arquivo do {project_selected} não foi encontrado.")
