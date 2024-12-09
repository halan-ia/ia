import streamlit as st
import os
import importlib.util

# Configuração do título
st.set_page_config(page_title="Home", layout="centered")

# Diretório onde os projetos estão localizados
scripts_dir = "pages/projeto1.py"

# Página inicial
st.title("Bem-vindo ao Hub de Projetos")
st.markdown("Escolha um dos projetos abaixo para começar:")

# Menu de navegação
project_options = {
    "Projeto 1": "projeto1",  # Nome do módulo, sem a extensão .py
    "Projeto 2": "projeto2",
}

# Seleção do projeto
project_selected = st.selectbox("Selecione um projeto:", ["Selecione"] + list(project_options.keys()))

if project_selected != "Selecione":
    script_name = project_options[project_selected]
    script_path = os.path.join(scripts_dir, f"{script_name}.py")
    
    # Verificando se o arquivo existe
    if os.path.exists(script_path):
        # Carregando o script como módulo
        spec = importlib.util.spec_from_file_location(script_name, script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Caso o módulo tenha funções específicas, você pode chamá-las assim:
        # module.main() ou qualquer outra função definida no script.
        
        st.success(f"Projeto {project_selected} carregado com sucesso!")
    else:
        st.error(f"Erro: O arquivo do {project_selected} não foi encontrado.")
