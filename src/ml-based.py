from utils import pd, verificar_dataset
from load import dt_mlb
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

dt_mlb = pd.read_csv("data/processed/store_product_reviews_ml_transformed.csv", sep = ';')
verificar_dataset("Dataset importado com sucesso.",dt_mlb,"Store Product")

print("Análise em andamento.")

# holdout method (split train/test)
x_train, x_test, y_train, y_test = train_test_split(dt_mlb['comentario'], dt_mlb['sentimento'], test_size=0.25, random_state=42, stratify=dt_mlb['sentimento'])

# transformar texto dos comentarios em numeros
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1,3),
    max_features=5000,
    min_df=2,
    max_df=0.9
)
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

# treinar e testar o modelo
modelo = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)
modelo.fit(x_train_vec, y_train)
modelo_preds = modelo.predict(x_test_vec)

# resultados
print("\nEstatísticas do Algoritmo Machine Learning Based")

acuracia = accuracy_score(y_test, modelo_preds)
cv_scores = cross_val_score(modelo, vectorizer.fit_transform(dt_mlb['comentario']), dt_mlb['sentimento'], cv=5, scoring='accuracy')
matriz = confusion_matrix(y_test, modelo_preds)

print("\nQuantidade de comentarios por cada sentimento antes da classificacao - DATASET DE TESTE:")
print(pd.Series(y_test).value_counts().sort_index())

print("\nQuantidade de comentarios por cada sentimento depois da classificacao - DATASET DE TESTE:")
print(pd.Series(modelo_preds).value_counts().sort_index())

print(f"\nAcurácia: {acuracia:.3f} ({acuracia * 100:.1f}%)")
print(f"\nF1-Score: {f1_score(y_test, modelo_preds, average='weighted'):.3f}")
print(f"\nValidação Cruzada: {cv_scores.mean():.3f} (± {cv_scores.std():.3f})")
print(f"Pontuacoes da Validação Cruzada: {[f'{score:.3f}' for score in cv_scores]}")

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