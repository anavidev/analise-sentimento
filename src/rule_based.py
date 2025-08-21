import numpy as np

from load import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix, f1_score

analisador = SentimentIntensityAnalyzer()

dt = pd.read_csv("data/processed/store_product_reviews_transformed.csv", sep = ',')
verificar_dataset("Dataset importado com sucesso.",dt,"Teste")

print("Análise em andamento.")

# aplicacao do analisador em cada comentario
dt['pontuacao_vader'] = dt['comentario'].apply(lambda texto: analisador.polarity_scores(texto))

# extrair compound de cada comentario e classifica-lo
dt['compound_vader'] = dt['pontuacao_vader'].apply(lambda x: x['compound'])
dt['sentimento_novo'] = dt['compound_vader'].apply(
    lambda x: 1 if x >= 0.05
    else -1 if x <= -0.05
    else 0
)

# resultados
print("\nEstatísticas do Algoritmo Rule-Based\n")

print("Acuracia:")
acuracia = np.mean(dt['sentimento_novo'] == dt['sentimento'])
print(f'{acuracia:.4f} = {acuracia * 100:.2f}%')

print("\nF1-Score:")
print(f"{f1_score(dt['sentimento'], dt['sentimento_novo'], average='weighted'):.3f}")

print("\nMatriz de Confusão:")
matriz = confusion_matrix(dt['sentimento'], dt['sentimento_novo'])
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

print("\nQuantidade de comentarios por cada sentimento antes da classificacao:")
print(dt.groupby('sentimento')['compound_vader'].count())

print("\nQuantidade de comentarios por cada sentimento depois da classificacao:")
print(dt.groupby('sentimento_novo')['compound_vader'].count())

# carregar resultado
dt = excluir_coluna(dt,'pontuacao_vader',"Teste")
dt.to_csv('data/processed/store_product_reviews_rb.csv', index=False, sep=',')