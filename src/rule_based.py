import numpy as np
from sklearn.metrics import accuracy_score

from utils import pd, verificar_dataset
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix, f1_score

analisador = SentimentIntensityAnalyzer()

dt = pd.read_csv("data/processed/store_product_reviews_transformed.csv", sep = ';')
verificar_dataset("Dataset importado com sucesso.",dt,"Store Product")

print("Análise em andamento.")

# aplicacao do analisador em cada comentario
dt['pontuacao_vader'] = dt['comentario'].apply(lambda texto: analisador.polarity_scores(texto))

# extrair compound de cada comentario e classifica-lo
dt['compound_vader'] = dt['pontuacao_vader'].apply(lambda x: x['compound'])
dt['sentimento_pred'] = dt['compound_vader'].apply(
    lambda x: 1 if x >= 0.05
    else -1 if x <= -0.05
    else 0
)

# resultados
print("\nEstatísticas do Algoritmo Rule-Based\n")

print("Quantidade de comentarios por cada sentimento antes da classificacao - DATASET DE TESTE:")
print(dt.groupby('sentimento')['compound_vader'].count())

print("\nQuantidade de comentarios por cada sentimento depois da classificacao - DATASET DE TESTE:")
print(dt.groupby('sentimento_pred')['compound_vader'].count())

print("\nAcuracia:")
acuracia = accuracy_score(dt['sentimento'], dt['sentimento_pred'])
print(f'{acuracia:.4f} = {acuracia * 100:.2f}%')

print("\nF1-Score:")
print(f"{f1_score(dt['sentimento'], dt['sentimento_pred'], average='weighted'):.3f}")

print("\nMatriz de Confusão:")
matriz = confusion_matrix(dt['sentimento'], dt['sentimento_pred'])
print(f"""Valores Negativos Corretos = {matriz[0][0]}
Valores Neutros Corretos = {matriz[1][1]} 
Valores Positivos Corretos = {matriz[2][2]}\n
Valores Negativos classificados como Neutros = {matriz[0][1]}
Valores Negativos classificados como Positivos = {matriz[0][2]}
Valores Neutros classificados como Negativos = {matriz[1][0]}
Valores Neutros classificados como Positivos = {matriz[1][2]}
Valores Positivos classificados como Negativos = {matriz[2][0]}
Valores Positivos classificados como Neutros = {matriz[2][1]}
""")