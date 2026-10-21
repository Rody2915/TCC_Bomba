import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Métricas e Sinais", page_icon="📈", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_VISUALIZACOES = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "visualizacoes"))

st.title("📈 Análise Dinâmica do Sinal e Desempenho da IA")
st.markdown("---")

# ==============================================================================
# SEÇÃO 1: GRÁFICOS DINÂMICOS DO ARQUIVO ANEXADO
# ==============================================================================
st.header("📊 Diagnóstico Físico do Ensaio Atual")

# Verifica se o usuário já anexou um arquivo na página de Análise
if "sinal_bruto" in st.session_state:
    st.success(f"🔍 Exibindo gráficos do sinal dinâmico avaliado como: **{st.session_state['diagnostico_atual']}** (Severidade: {st.session_state['severidade_atual']})")
    
    sinal = st.session_state["sinal_bruto"]
    features = st.session_state["features_atuais"]
    fs = 20000
    
    # Cálculos dinâmicos da FFT para o gráfico na tela
    tempo = np.linspace(0, len(sinal)/fs, len(sinal))
    fft = np.fft.fft(sinal)
    frequencias = np.fft.fftfreq(len(sinal), d=1/fs)
    amplitudes = np.abs(fft)
    
    metade = len(amplitudes) // 2
    frequencias = frequencias[:metade]
    amplitudes = amplitudes[:metade]
    
    # Renderização dos gráficos do arquivo anexado via Matplotlib
    col_sinal1, col_sinal2 = st.columns(2)
    
    with col_sinal1:
        st.markdown("### 🕒 Forma de Onda no Domínio do Tempo")
        fig_tempo, ax_t = plt.subplots(figsize=(10, 4.5))
        ax_t.plot(tempo[:2000], sinal[:2000], color="#1f77b4", alpha=0.8) # Plota os primeiros 2000 pontos para não travar a tela
        ax_t.set_title("Sinal de Vibração Bruto (Truncado para Visualização)")
        ax_t.set_xlabel("Tempo (s)")
        ax_t.set_ylabel("Amplitude de Aceleração (g)")
        ax_t.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig_tempo)
        
    with col_sinal2:
        st.markdown("### 🌌 Espectro de Frequência (FFT)")
        fig_freq, ax_f = plt.subplots(figsize=(10, 4.5))
        ax_f.plot(frequencias[frequencias <= 2000], amplitudes[frequencias <= 2000], color="#ff7f0e")
        ax_f.set_title("Espectro de Amplitude Harmônica (Até 2000 Hz)")
        ax_f.set_xlabel("Frequência (Hz)")
        ax_f.set_ylabel("Amplitude")
        ax_f.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig_freq)
        
    # Tabela com as 11 características calculadas na hora para esse arquivo
    st.markdown("### 📋 Vetor de Características Calculado para este Arquivo")
    st.dataframe(features, use_container_width=True)

else:
    st.info("💡 **Nenhum arquivo foi anexado ainda.** Vá até a página **1_Analise** no menu lateral, faça o upload de um arquivo CSV e clique em processar para liberar os gráficos em tempo real desta seção.")

st.markdown("---")

# ==============================================================================
# SEÇÃO 2: MÉTRICAS HISTÓRICAS FIXAS DO MODELO (EXIGÊNCIA DA BANCA)
# ==============================================================================
st.header("📈 Desempenho Estatístico Global do Modelo (Validação Cruzada)")
st.write("Métricas de assertividade calculadas sobre a base de dados de teste (20% do dataset global de 2.650 amostras).")

col1, col2 = st.columns(2)
with col1:
    st.metric(label="Acurácia Baseline (Apenas Domínio do Tempo)", value="54.72%")
with col2:
    st.metric(label="Acurácia Otimizada Multicanais (Random Forest)", value="80.38%", delta="+25.66% de Ganho Real")

col_grafico1, col_grafico2 = st.columns(2)
caminho_matriz = os.path.join(DIR_VISUALIZACOES, "matriz_confusao.png")
caminho_importancia = os.path.join(DIR_VISUALIZACOES, "importancia_features.png")

with col_grafico1:
    st.markdown("### 📊 Matriz de Confusão Global do Modelo")
    if os.path.exists(caminho_matriz):
        st.image(caminho_matriz, caption="Mapeamento de Acertos e Erros Fixo do Treinamento", use_container_width=True)
    else:
        st.warning("⚠️ Imagem 'matriz_confusao.png' não encontrada.")
        
with col_grafico2:
    st.markdown("### 📈 Relevância Estrutural das Variáveis (Gini Importance)")
    if os.path.exists(caminho_importancia):
        st.image(caminho_importancia, caption="Peso fixo de decisão das 700 Árvores", use_container_width=True)
    else:
        st.warning("⚠️ Imagem 'importancia_features.png' não encontrada.")
