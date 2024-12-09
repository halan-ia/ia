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
    "Projeto 1": "pagina1",  # Nome do arquivo sem a extensão .py
    "Projeto 2": "pagina2",
}

# Seleção do projeto
project_selected = st.selectbox("Selecione um projeto:", ["Selecione"] + list(project_options.keys()))

if project_selected != "Selecione":
    script_name = project_options[project_selected]  # Nome do arquivo sem extensão
    script_path = os.path.join(scripts_dir, f"{script_name}.py")  # Adiciona a extensão .py ao nome do arquivo
    
    # Verificando se o arquivo existe
    if os.path.exists(script_path):
        # Carregando e executando o script
        with open(script_path, "r") as file:
            code = file.read()
            exec(code)  # Executa o código Python do script selecionado
    else:
        st.error(f"Erro: O arquivo do {project_selected} não foi encontrado.")
        st.write(f"Caminho procurado: {script_path}")  # Exibe o caminho para depuração
