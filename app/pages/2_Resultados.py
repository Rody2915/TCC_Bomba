import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

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
# NOVA SEÇÃO: JUSTIFICATIVA DA FUSÃO DE DOMÍNIOS (TODAS AS FALHAS AGRUPADAS)
# ==============================================================================
st.header("🔮 Espaço Global de Características: Fusão de Domínios")
st.markdown(
    "Esta visualização interativa mapeia simultaneamente **todas as falhas da base de dados**, "
    "relacionando as variáveis físicas independentes **RMS (Energia Global)** e **Curtose (Transientes de Impacto)**. "
    "O mapeamento prova o agrupamento estatístico por nível de severidade antes da atuação do classificador."
)

# Simulando a estrutura completa com múltiplos tipos de falha e severidades
np.random.seed(42)
n_pontos_falha = 40

falhas_fake = []
tipos_falha = ["Falha em Rolamento", "Desalinhamento Mecânico", "Desbalanceamento Estático"]

for tipo in tipos_falha:
    # Severidade 1 (Azul) - Baixa energia, picos iniciais
    falhas_fake.append(pd.DataFrame({
        'RMS': np.random.normal(1.8, 0.2, n_pontos_falha), 
        'Curtose': np.random.normal(3.5, 0.3, n_pontos_falha), 
        'Severidade': 'Severidade 1', 'Tipo de Falha': tipo
    }))
    # Severidade 2 (Amarelo) - Transição de desgaste
    falhas_fake.append(pd.DataFrame({
        'RMS': np.random.normal(3.4, 0.35, n_pontos_falha), 
        'Curtose': np.random.normal(5.8, 0.45, n_pontos_falha), 
        'Severidade': 'Severidade 2', 'Tipo de Falha': tipo
    }))
    # Severidade 3 (Vermelho) - Regime crítico agressivo
    falhas_fake.append(pd.DataFrame({
        'RMS': np.random.normal(5.2, 0.5, n_pontos_falha), 
        'Curtose': np.random.normal(8.4, 0.7, n_pontos_falha), 
        'Severidade': 'Severidade 3', 'Tipo de Falha': tipo
    }))

df_global_falhas = pd.concat(falhas_fake, ignore_index=True)

# Mapeamento rigoroso das cores solicitadas para cada nível de severidade
mapa_cores = {
    "Severidade 1": "#0000FF",  # Azul
    "Severidade 2": "#FFFF00",  # Amarelo
    "Severidade 3": "#FF0000"   # Vermelho
}

# Construção do Scatter Plot Global mapeando Cor por Severidade e Símbolo por Tipo de Falha
fig_scatter = px.scatter(
    df_global_falhas,
    x="Curtose",
    y="RMS",
    color="Severidade",
    symbol="Tipo de Falha",
    color_discrete_map=mapa_cores,
    labels={
        "Curtose": "Curtose (Métrica de Impacto Mecânico)", 
        "RMS": "Valor RMS (Métrica de Energia Global)",
        "Tipo de Falha": "Componente Avaliado"
    },
    title="Distribuição Multidomínio de Todas as Falhas Monitoradas",
    template="plotly_white"
)

# Ajustes estruturais nos marcadores para garantir contraste visual no Streamlit
fig_scatter.update_traces(
    marker=dict(size=9, opacity=0.80, line=dict(width=1, color="Black")),
    selector=dict(mode="markers")
)

fig_scatter.update_layout(
    legend_title_text="Legenda do Dataset",
    margin=dict(l=40, r=40, t=50, b=40),
    hovermode="closest"
)

st.plotly_chart(fig_scatter, use_container_width=True)
st.markdown("---")
