from load import *
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

dt = pd.read_csv("data/processed/store_product_reviews_ml_transformed.csv", sep = ';')
verificar_dataset("Dataset importado com sucesso.",dt,"Store Product")

print("Análise em andamento.")

# holdout method (split train/test)
x_train, x_test, y_train, y_test = train_test_split(dt['comentario'], dt['sentimento'], test_size=0.2, random_state=42)

# transformar texto dos comentarios em numeros
vectorizer = TfidfVectorizer(stop_words='english',ngram_range=(1,3), max_features=5000, min_df=2)
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

# treinar e testar o modelo
modelo = LogisticRegression(max_iter=1000, C=1)
modelo.fit(x_train_vec, y_train)
modelo_preds = modelo.predict(x_test_vec)

# resultados
print("\nEstatísticas do Algoritmo Machine Learning Based")

print("\nQuantidade de comentarios por cada sentimento antes da classificacao - DATASET DE TESTE:")
print(pd.Series(y_test).value_counts().sort_index())

print("\nQuantidade de comentarios por cada sentimento depois da classificacao - DATASET DE TESTE:")
print(pd.Series(modelo_preds).value_counts().sort_index())

print(f"\nAcuracia:")
acuracia = accuracy_score(y_test, modelo_preds)
print(f'{acuracia:.4f} = {acuracia * 100:.2f}%')

print("\nF1-Score:")
print(f"{f1_score(y_test, modelo_preds, average='weighted'):.3f}")

print("\nMatriz de Confusão:")
matriz = confusion_matrix(y_test, modelo_preds)
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