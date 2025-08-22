from utils import pd, verificar_dataset
from load import dt_rb
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix, f1_score

analisador = SentimentIntensityAnalyzer()

dt_rb = pd.read_csv("data/processed/store_product_reviews_transformed.csv", sep = ';')
verificar_dataset("Dataset importado com sucesso.",dt_rb,"Store Product")

print("Análise em andamento.")

# aplicacao do analisador em cada comentario
dt_rb['pontuacao_vader'] = dt_rb['comentario'].apply(lambda texto: analisador.polarity_scores(texto))

# extrair compound de cada comentario e classifica-lo
dt_rb['compound_vader'] = dt_rb['pontuacao_vader'].apply(lambda x: x['compound'])
dt_rb['sentimento_pred'] = dt_rb['compound_vader'].apply(
    lambda x: 1 if x >= 0.05
    else -1 if x <= -0.05
    else 0
)

# resultados
print("\nEstatísticas do Algoritmo Rule-Based\n")

acuracia = accuracy_score(dt_rb['sentimento'], dt_rb['sentimento_pred'])
matriz = confusion_matrix(dt_rb['sentimento'], dt_rb['sentimento_pred'])

print("Quantidade de comentarios por cada sentimento antes da classificacao - DATASET DE TESTE:")
print(dt_rb.groupby('sentimento')['compound_vader'].count())

print("\nQuantidade de comentarios por cada sentimento depois da classificacao - DATASET DE TESTE:")
print(dt_rb.groupby('sentimento_pred')['compound_vader'].count())

print(f"\nAcurácia: {acuracia:.3f} ({acuracia * 100:.1f}%)")
print(f"\nF1-Score: {f1_score(dt_rb['sentimento'], dt_rb['sentimento_pred'], average='weighted'):.3f}")

print("\nMatriz de Confusão:")
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