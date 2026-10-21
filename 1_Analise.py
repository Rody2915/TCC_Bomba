import streamlit as st
import os
import joblib
import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from scipy.signal import find_peaks

# Configuração da página interna
st.set_page_config(page_title="Análise de Vibração", page_icon="📊", layout="wide")

# Caminhos dinâmicos para carregar o modelo de IA
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
caminho_modelo = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "modelo", "modelo_rf_v2.pkl"))

@st.cache_resource
def carregar_modelo_ia():
    if os.path.exists(caminho_modelo):
        return joblib.load(caminho_modelo)
    return None

modelo = carregar_modelo_ia()

# Função integrada de extração de características (Valores matemáticos fixados em segundo plano)
def extrair_features_geral(uploaded_file, fs, freq_corte, canal_alvo, prom, dist):
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)
    if canal_alvo not in df.columns:
        raise ValueError(f"A coluna '{canal_alvo}' não foi encontrada no arquivo enviado.")
        
    sinal = df[canal_alvo]
    
    # --- PROCESSAMENTO FFT ---
    fft = np.fft.fft(sinal)
    frequencias = np.fft.fftfreq(len(sinal), d=1/fs)
    amplitudes = np.abs(fft)
    
    metade = len(amplitudes) // 2
    frequencias = frequencias[:metade]
    amplitudes = amplitudes[:metade]
    
    mascara = frequencias <= freq_corte
    frequencias_baixas = frequencias[mascara]
    amplitudes_baixas = amplitudes[mascara]
    
    picos, _ = find_peaks(amplitudes_baixas, prominence=prom, distance=dist)
    amplitudes_picos = amplitudes_baixas[picos]
    ordem = np.argsort(amplitudes_picos)[::-1]
    
    if len(ordem) >= 3:
        freq1 = float(frequencias_baixas[picos[ordem[0]]])
        amp1 = float(amplitudes_baixas[picos[ordem[0]]])
        freq2 = float(frequencias_baixas[picos[ordem[1]]])
        amp2 = float(amplitudes_baixas[picos[ordem[1]]])
        freq3 = float(frequencias_baixas[picos[ordem[2]]])
        amp3 = float(amplitudes_baixas[picos[ordem[2]]])
    else:
        freq1 = amp1 = freq2 = amp2 = freq3 = amp3 = np.nan        
        
    # --- ESTATÍSTICA (Tempo) ---
    rms = np.sqrt(np.mean(sinal ** 2))
    pico = np.max(np.abs(sinal))
    desvio = np.std(sinal)
    curtose_val = kurtosis(sinal, fisher=False)
    fator_crista = pico / rms
    
    return pd.DataFrame([{
        "RMS": rms, "Pico": pico, "Devio": desvio, "Curtose": curtose_val, "Fator_Crista": fator_crista,
        "Freq_Pico_1": freq1, "Amp_Pico_1": amp1, "Freq_Pico_2": freq2, "Amp_Pico_2": amp2, "Freq_Pico_3": freq3, "Amp_Pico_3": amp3
    }])

# Dicionário de falhas técnico expandido com as cores e severidades corretas
DICON_FALHAS = {
    "healthy": {"nome": "Condição Saudável", "desc": "Equipamento operando dentro dos padrões ideais de projeto.", "cor": "green"},
    "healthy noise": {"nome": "Condição Saudável (Com Ruído)", "desc": "Assinatura operacional aceitável com presença de ruído de fundo.", "cor": "green"},
    "new motor": {"nome": "Motor Novo (Amaciamento)", "desc": "Componente em conformação inicial sem defeitos estruturais.", "cor": "green"},
    "bearing bpfi": {"nome": "Falha no Rolamento (Pista Interna)", "desc": "Defeito concentrado na pista interna dos elementos rotativos.", "cor": "red"},
    "bearing bpfo": {"nome": "Falha no Rolamento (Pista Externa)", "desc": "Defeito concentrado na pista externa do rolamento.", "cor": "red"},
    "bearing bsf": {"nome": "Falha no Rolamento (Esfera)", "desc": "Defeito micro-localizado nos elementos rotantes (esferas).", "cor": "orange"},
    "bearing contaminated": {"nome": "Rolamento Contaminado", "desc": "Falta de lubrificação eficiente ou infiltração de abrasivos.", "cor": "orange"},
    "bearing pump": {"nome": "Falha Geral no Rolamento da Bomba", "desc": "Avaria identificada no conjunto de suporte do eixo hidráulico.", "cor": "red"},
    "broken rotor bar": {"nome": "Barra Quebrada do Rotor", "desc": "Falha mecânica-elétrica nas barras de indução do rotor.", "cor": "red"},
    "impeller": {"nome": "Dano no Impulsor da Bomba", "desc": "Desgaste, cavitação severa ou quebra nas pás hidráulicas.", "cor": "red"},
    "loose foot motor": {"nome": "Pé Frouxo no Motor", "desc": "Frouxidão mecânica identificada na fixação da base do motor.", "cor": "orange"},
    "loose foot pump": {"nome": "Pé Frouxo na Bomba", "desc": "Instabilidade mecânica na fixação estrutural da voluta da bomba.", "cor": "orange"},
    "soft foot": {"nome": "Pé Manco (Soft Foot)", "desc": "Problema crítico de nivelamento ou distorção geométrica na base.", "cor": "orange"},
    "stator short": {"nome": "Curto-Circuito no Estator", "desc": "Falha severa de isolamento nas bobinas elétricas do estator.", "cor": "red"}
}

def tratar_classe_label(classe_bruta):
    classe_limpa = str(classe_bruta).lower().strip()
    for chave in DICON_FALHAS.keys():
        if chave in classe_limpa:
            # Captura qualquer número de severidade presente na string (ex: 'impeller 3' -> '3')
            severidade = "".join(filter(str.isdigit, classe_limpa))
            return chave, severidade if severidade else "Única"
    return "desconhecida", "Desconhecida"

# --- APENAS OS 3 INPUTS SOLICITADOS NA BARRA LATERAL ---
st.sidebar.title("⚙️ Parâmetros Operacionais")

st.sidebar.markdown("### Configurações do Ativo")
motor_id = st.sidebar.text_input("Identificação do Motor/Bomba", value="Motor-2")
capacidade_op = st.sidebar.slider("Capacidade Operacional (%)", min_value=1, max_value=100, value=75)

st.sidebar.markdown("### Posição de Medição")
posicao_sensor = st.sidebar.selectbox("Localização do Acelerômetro", [
    "1 - Carcaça do Motor (Lado NDE - Traseiro)", 
    "2 - Mancal do Motor (Lado DE - Dianteiro)", 
    "3 - Mancal do Motor (Direção Axial)", 
    "4 - Carcaça do Mancal da Bomba (Lado Acoplamento)", 
    "5 - Carcaça do Mancal da Bomba (Lado Impulsor)"
], index=3)

# Mapeamento do nome escolhido para buscar termos no arquivo CSV do usuário
MAPA_SUFIXOS = {
    "1 - Carcaça do Motor (Lado NDE - Traseiro)": ["ch1", "canal1", "canal_1", "motor_traseiro"], 
    "2 - Mancal do Motor (Lado DE - Dianteiro)": ["ch2", "canal2", "canal_2", "motor_dianteiro"],
    "3 - Mancal do Motor (Direção Axial)": ["ch3", "canal3", "canal_3", "axial"], 
    "4 - Carcaça do Mancal da Bomba (Lado Acoplamento)": ["ch4", "canal4", "canal_4", "bomba_acoplamento"], 
    "5 - Carcaça do Mancal da Bomba (Lado Impulsor)": ["ch5", "canal5", "canal_5", "bomba_impulsor"]
}
lista_sufixos_alvo = MAPA_SUFIXOS[posicao_sensor]

# Mapeamento do identificador numérico exigido pela IA universal
MAPA_ID_CANAIS = {
    "1 - Carcaça do Motor (Lado NDE - Traseiro)": 1, 
    "2 - Mancal do Motor (Lado DE - Dianteiro)": 2,
    "3 - Mancal do Motor (Direção Axial)": 3, 
    "4 - Carcaça do Mancal da Bomba (Lado Acoplamento)": 4, 
    "5 - Carcaça do Mancal da Bomba (Lado Impulsor)": 5
}
id_canal_ia = MAPA_ID_CANAIS[posicao_sensor]

# Valores fixos padronizados do modelo de IA
FS_PADRAO = 20000
FREQ_CORTE_PADRAO = 2000
PROMINENCE_PADRAO = 300   # Sincronizado com o novo extrair_features.py (Opção A)
DISTANCE_PADRAO = 300      # Sincronizado com o novo extrair_features.py (Opção A)

# --- CORPO DA PÁGINA ---
st.title("📊 Análise de Sinais Inteligente")
st.markdown("---")

if modelo is None:
    st.error("Não é possível prosseguir: O arquivo 'modelo_rf_v2.pkl' não foi carregado corretamente.")
else:
    st.info(f"📍 Ponto selecionado: **{posicao_sensor}** | Código interno repassado à IA: `Canal_Sensor = {id_canal_ia}`")
    arquivo_upload = st.file_uploader("Faça o upload do arquivo CSV bruto de vibração", type=["csv"])
    
    if arquivo_upload is not None:
        if st.button("🚀 Iniciar Processamento Adaptativo"):
            with st.spinner("Executando Engenharia de Atributos e Consultando a IA..."):
                try:
                    # Garante que o cursor de leitura comece estritamente no início do arquivo
                    arquivo_upload.seek(0)
                    
                    # Leitura única dos cabeçalhos para mapeamento dinâmico
                    df_header = pd.read_csv(arquivo_upload, nrows=0)
                    canal_final = None
                    
                    for col_real in df_header.columns:
                        col_limpa = str(col_real).lower().strip()
                        if any(sufixo in col_limpa for sufixo in lista_sufixos_alvo):
                            canal_final = col_real
                            break
                    
                    if canal_final is None:
                        if id_canal_ia < len(df_header.columns):
                            canal_final = df_header.columns[id_canal_ia]
                        else:
                            canal_final = df_header.columns[-1]
                    
                    # RESET CRUCIAL: Reseta o cursor novamente para a função extrair_features_geral conseguir ler os dados brutos
                    arquivo_upload.seek(0)
                    
                    # Extração das características estatísticas e espectrais
                    X_pred = extrair_features_geral(
                        arquivo_upload, 
                        fs=FS_PADRAO, 
                        freq_corte=FREQ_CORTE_PADRAO, 
                        canal_alvo=canal_final, 
                        prom=PROMINENCE_PADRAO, 
                        dist=DISTANCE_PADRAO
                    )
                    
                    # Inclusão da coluna de sensor exigida pela IA universal
                    X_pred["Canal_Sensor"] = id_canal_ia
                    
                    # Realinha exatamente na ordem exigida pelas 700 árvores do Random Forest
                    X_pred = X_pred[[
                        "Canal_Sensor", "RMS", "Pico", "Devio", "Curtose", "Fator_Crista", 
                        "Freq_Pico_1", "Amp_Pico_1", "Freq_Pico_2", "Amp_Pico_2", "Freq_Pico_3", "Amp_Pico_3"
                    ]]
                    
                    # Predição do Random Forest Otimizado
                    resultado_ia = modelo.predict(X_pred)
                    predicao_str = str(resultado_ia).lower().strip()
                    
                    # Interpretação do rótulo e severidade
                    chave_falha, severidade_falha = tratar_classe_label(predicao_str)
                    info_falha = DICON_FALHAS.get(chave_falha, {"nome": "Desconhecida", "desc": "Padrão de falha não catalogado.", "cor": "orange"})
                    
                    # RESET ANTES DE COMPARTILHAR: Reseta o ponteiro para ler o sinal bruto completo para o gráfico dinâmico
                    arquivo_upload.seek(0)
                    st.session_state["sinal_bruto"] = pd.read_csv(arquivo_upload)[canal_final].values
                    st.session_state["features_atuais"] = X_pred
                    st.session_state["diagnostico_atual"] = info_falha["nome"]
                    st.session_state["severidade_atual"] = severidade_falha

                    # Painel de Diagnóstico na Tela do Usuário
                    st.markdown("### Status do Diagnóstico do Ativo:")
                    if info_falha["cor"] == "green": 
                        st.balloons()
                        st.success(f"## **{info_falha['nome']}** (Severidade: {severidade_falha})")
                    elif info_falha["cor"] == "orange": 
                        st.warning(f"## **{info_falha['nome']}** (Severidade: {severidade_falha})")
                    else: 
                        st.error(f"## **{info_falha['nome']}** (Severidade: {severidade_falha})")
                    
                    st.write(f"**Ativo Monitorado:** *{motor_id}* operando a *{capacidade_op}%* de carga.")
                    st.write(f"**Análise da Localização:** Diagnóstico gerado com base no sensor posicionado em: *{posicao_sensor}*.")
                    st.write(f"**Laudo Mecânico Descritivo:** {info_falha['desc']}")
                    
                    with st.expander("🔍 Detalhes Técnicos do Vetor de Atributos (Auditoria da Banca)"):
                        st.write("Vetor de características extraído e processado na memória RAM:")
                        st.dataframe(X_pred)
                        
                except Exception as e:
                    st.error(f"Erro durante o processamento do arquivo: {e}")

