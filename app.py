import streamlit as st
import os
import joblib

# Configuração global da aplicação
st.set_page_config(page_title="TCC Bombas RF", page_icon="⚙️", layout="wide")

# Caminhos dinâmicos para verificar o modelo de IA
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
caminho_modelo = os.path.abspath(os.path.join(BASE_DIR, "..", "modelo", "modelo_rf_v2.pkl"))

st.title("⚙️ Sistema Universal para Diagnóstico de Sistemas Hidráulicos")
st.subheader("Plataforma Flexível Voltada à Manutenção Preditiva Otimizada")
st.markdown("---")

st.write("Bem-vindo ao sistema de diagnóstico automatizado desenvolvido para o TCC de Engenharia Mecânica.")
st.write("Este software processa sinais dinâmicos de vibração em tempo real e emprega um algoritmo de inteligência artificial treinado para classificar falhas mecânicas, hidráulicas e elétricas.")

if os.path.exists(caminho_modelo):
    st.success("✅ Cérebro da Inteligência Artificial (Modelo Random Forest) detectado no servidor e pronto para operação.")
else:
    st.error("❌ Arquivo 'modelo_rf_v2.pkl' não foi encontrado na pasta 'modelo/'. Por favor, execute o script de treinamento primeiro.")

st.info("💡 Utilize o menu lateral esquerdo para navegar pelas páginas de **Análise de Vibração** e **Resultados Técnicos**.")
