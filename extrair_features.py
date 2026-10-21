import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy.stats import kurtosis
from scipy.signal import find_peaks

# ==============================================================================
# 1. CONFIGURAÇÃO DE DIRETÓRIOS DO SEU PC
# ==============================================================================
diretorio_dataset = r"C:\Users\rodri\OneDrive\Desktop\TCC_Bombas_RF\dados_brutos\Dataset\Vibration"
diretorio_modelo_saida = r"C:\Users\rodri\OneDrive\Desktop\TCC_Bombas_RF\modelo"
diretorio_visualizacoes = r"C:\Users\rodri\OneDrive\Desktop\TCC_Bombas_RF\visualizacoes"

# Garante que as pastas de saída existam no Windows
os.makedirs(diretorio_modelo_saida, exist_ok=True)
os.makedirs(diretorio_visualizacoes, exist_ok=True)

# ==============================================================================
# 2. FUNÇÃO MATEMÁTICA DE EXTRAÇÃO DE CARACTERÍSTICAS (OTIMIZAÇÃO DA FFT)
# ==============================================================================
def extrair_features_matriz(caminho_arquivo):
    df = pd.read_csv(caminho_arquivo)
    resultados = []
    fs = 20000

    # Percorre as 15 janelas temporais de medição do arquivo
    for coluna in df.columns[1:]:
        sinal = df[coluna]
        
        # --- PROCESSAMENTO FFT (Domínio da Frequência) ---
        fft = np.fft.fft(sinal)
        frequencias = np.fft.fftfreq(len(sinal), d=1/fs)
        amplitudes = np.abs(fft)
        
        metade = len(amplitudes) // 2
        frequencias = frequencias[:metade]
        amplitudes = amplitudes[:metade]
        
        mascara = frequencias <= 2000
        frequencias_baixas = frequencias[mascara]
        amplitudes_baixas = amplitudes[mascara]
        
        # OTIMIZAÇÃO A: Parâmetros refinados para mitigar NaNs e capturar picos finos de rolamento
        picos, _ = find_peaks(amplitudes_baixas, prominence=300, distance=300)
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
        
        # --- ESTATÍSTICA MATEMÁTICA (Domínio do Tempo) ---
        rms = np.sqrt(np.mean(sinal ** 2))
        pico = np.max(np.abs(sinal))
        desvio = np.std(sinal)
        curtose_val = kurtosis(sinal, fisher=False)
        fator_crista = pico / rms
        
        resultados.append({
            "RMS": rms, "Pico": pico, "Devio": desvio, # Mantido 'Devio' se sua tabela usar essa grafia
            "Curtose": curtose_val, "Fator_Crista": fator_crista,
            "Freq_Pico_1": freq1, "Amp_Pico_1": amp1,
            "Freq_Pico_2": freq2, "Amp_Pico_2": amp2,
            "Freq_Pico_3": freq3, "Amp_Pico_3": amp3
        })

    return pd.DataFrame(resultados)

# ==============================================================================
# 3. CONSTRUÇÃO AUTOMATIZADA COMPLETA DO DATASET (24 CLASSES DE SEVERIDADE)
# ==============================================================================
todas_as_classes = []

MAPEAMENTO_PASTAS = {
    # Falhas de Pista Interna (BPFI)
    "Motor-2/75/bearing bpfi 1": "bearing bpfi 1",
    "Motor-2/75/bearing bpfi 2": "bearing bpfi 2",
    "Motor-2/75/bearing bpfi 3": "bearing bpfi 3",

    # Falhas de Pista Externa (BPFO)
    "Motor-2/75/bearing bpfo 1": "bearing bpfo 1",
    "Motor-2/75/bearing bpfo 2": "bearing bpfo 2",
    "Motor-2/75/bearing bpfo 3": "bearing bpfo 3",
    
    # Outras Falhas de Rolamento e Elétricas
    "Motor-2/75/bearing bsf": "bearing bsf",
    "Motor-2/75/bearing contaminated": "bearing contaminated",
    "Motor-2/75/broken rotor bar": "broken rotor bar",

    # Falhas Gerais de Rolamento da Bomba
    "Motor-2/75/bearing pump 1": "bearing pump 1",
    "Motor-2/75/bearing pump 2": "bearing pump 2",
    "Motor-2/75/bearing pump 3": "bearing pump 3",

    # Condições Saudáveis Operacionais
    "Motor-2/75/healthy 1": "healthy",
    "Motor-2/75/healthy 2": "healthy",
    "Motor-2/75/healthy 3": "healthy",
    "Motor-2/75/healthy noise": "healthy noise",
    
    # Falhas de Cavitação/Impulsor
    "Motor-2/75/impeller 1": "impeller 1",
    "Motor-2/75/impeller 2": "impeller 2",
    "Motor-2/75/impeller 3": "impeller 3",
    
    # Falhas de Fixação e Estrutura Estrutural
    "Motor-2/75/loose foot motor": "loose foot motor",
    "Motor-2/75/loose foot pump": "loose foot pump",
    "Motor-2/75/new motor": "new motor",
    "Motor-2/75/soft foot 1": "soft foot 1",
    "Motor-2/75/soft foot 2": "soft foot 2",

    # Curto-Circuito Elétrico no Estator
    "Motor-2/75/stator short 1": "stator short 1",
    "Motor-2/75/stator short 2": "stator short 2"
}

print("🚀 Iniciando varredura e extração MULTICANAIS global (Canais 1 a 5)...")

MAPA_NUMERICO_CANAIS = {
    "-ch1.csv": 1, "-ch2.csv": 2, "-ch3.csv": 3, "-ch4.csv": 4, "-ch5.csv": 5
}

for subpasta, label_classe in MAPEAMENTO_PASTAS.items():
    caminho_completo = os.path.join(diretorio_dataset, subpasta)
    
    if os.path.exists(caminho_completo):
        print(f"📁 Coletando assinaturas de vibração da condição: {label_classe}...")
        todos_arquivos = os.listdir(caminho_completo)
        
        for sufixo_ch, id_canal in MAPA_NUMERICO_CANAIS.items():
            arquivos_canal = [f for f in todos_arquivos if f.endswith(sufixo_ch)]
            
            for arquivo_csv in arquivos_canal:
                caminho_arquivo = os.path.join(caminho_completo, arquivo_csv)
                try:
                    df_features = extrair_features_matriz(caminho_arquivo)
                    df_features["Canal_Sensor"] = id_canal
                    df_features["Classe"] = label_classe
                    todas_as_classes.append(df_features)
                except Exception as e:
                    print(f"   ⚠️ Erro ao processar {arquivo_csv} (Canal {id_canal}): {e}")
    else:
        print(f"   ❌ Diretório ausente ou não encontrado: {caminho_completo}")

if len(todas_as_classes) == 0:
    print("\n❌ CRÍTICO: Nenhum arquivo foi minerado. Verifique a estrutura de pastas.")
    exit()

base_final = pd.concat(todas_as_classes, ignore_index=True)
nome_planilha_completa = os.path.join(diretorio_modelo_saida, "..", "base_features_completa.xlsx")
base_final.to_excel(nome_planilha_completa, index=False)
print(f"\n✅ MATRIZ GLOBAL GERADA: {len(base_final)} amostras salvas em 'base_features_completa.xlsx'!")

# ==============================================================================
# 4. RANDOM FOREST COM HIPERPARÂMETROS OTIMIZADOS VIA ENTROPIA
# ==============================================================================
X = base_final[[
    "Canal_Sensor", "RMS", "Pico", "Devio", "Curtose", "Fator_Crista",
    "Freq_Pico_1", "Amp_Pico_1", "Freq_Pico_2", "Amp_Pico_2", "Freq_Pico_3", "Amp_Pico_3"
]]
y = base_final["Classe"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n🧠 Treinando o Classificador Random Forest Otimizado (700 árvores via Entropy)...")
# OTIMIZAÇÃO A: Mudança de algoritmo estrutural para focar em limites finos de severidade
modelo = RandomForestClassifier(
    n_estimators=700, 
    criterion="entropy", 
    max_features="sqrt",
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
modelo.fit(X_train, y_train)
print("✅ Inteligência Artificial treinada com sucesso!")

# Avaliação de Métricas para a Banca do TCC
predicoes = modelo.predict(X_test)
acuracia = accuracy_score(y_test, predicoes)
print(f"\n📊 NOVA ACURÁCIA GLOBAL DO MODELO: {acuracia * 100:.2f}%")
print("\n📋 Relatório Técnico de Classificação:\n")
print(classification_report(y_test, predicoes))

# ==============================================================================
# 5. GERAÇÃO E SALVAMENTO DE GRÁFICOS ATUALIZADOS
# ==============================================================================
matriz = confusion_matrix(y_test, predicoes)
plt.figure(figsize=(15, 12))
sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", xticklabels=modelo.classes_, yticklabels=modelo.classes_)
plt.title("Matriz de Confusão Otimizada - Random Forest")
plt.xlabel("Classe Prevista")
plt.ylabel("Classe Real")
caminho_salvar_matriz = os.path.join(diretorio_visualizacoes, "matriz_confusao.png")
plt.savefig(caminho_salvar_matriz, dpi=300, bbox_inches='tight')
plt.close()

importancias = pd.DataFrame({"Variavel": X.columns, "Importancia": modelo.feature_importances_})
importancias = importancias.sort_values(by="Importancia", ascending=False)
importancias.to_excel(os.path.join(diretorio_visualizacoes, "importancia_variaveis.xlsx"), index=False)

plt.figure(figsize=(10, 6))
# Correção do Future Warning do Seaborn
sns.barplot(x="Importancia", y="Variavel", data=importancias, hue="Variavel", palette="viridis", legend=False)
plt.title("Grau de Relevância de Atributos - Random Forest Otimizado")
plt.xlabel("Peso de Importância (Entropy)")
plt.ylabel("Atributos Extraídos")
caminho_salvar_features = os.path.join(diretorio_visualizacoes, "importancia_features.png")
plt.savefig(caminho_salvar_features, dpi=300, bbox_inches='tight')
plt.close()
print("📊 Gráficos de desempenho atualizados salvos na pasta 'visualizacoes/'!")

# ==============================================================================
# 6. EXPORTAÇÃO DO CÉREBRO DA IA ATUALIZADO (.PKL COMPACTADO)
# ==============================================================================
caminho_final_pkl = os.path.join(diretorio_modelo_saida, "modelo_rf_v2.pkl")

# OTIMIZAÇÃO CRÍTICA: compress=3 compacta o arquivo para caber no limite do GitHub de 100MB
joblib.dump(modelo, caminho_final_pkl, compress=3) 

print(f"\n💾 NOVO CÉREBRO DA IA SALVO COM SUCESSO EM: {caminho_final_pkl}")
print("O sistema está pronto para rodar de forma universal!\n")

