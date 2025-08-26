from utils.utils import pd, verificar_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

def analisar_ml_based():
    dt_mlb = pd.read_csv("data/processed/store_product_reviews_ml_transformed.csv", sep = ';')
    verificar_dataset("Dataset importado com sucesso.",dt_mlb,"ML-Based")

    print("Analise em andamento.")

    # holdout method (split treino/test)
    x_treino, x_teste, y_treino, y_teste = train_test_split(
        dt_mlb['comentario'], 
        dt_mlb['sentimento'], 
        test_size=0.25, 
        random_state=42, 
        stratify=dt_mlb['sentimento']
    )

    # pipeline para aplicar tf-idf e configurar modelo de ml
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(
            stop_words='english',
            ngram_range=(1,2),
            max_features=15000,
            min_df=2,
            max_df=0.8
        )),
        ('classifier', LogisticRegression(
            max_iter=1000,
            class_weight='balanced'
        ))
    ])

    # treinar e testar o modelo
    pipeline.fit(x_treino, y_treino)
    modelo_preds = pipeline.predict(x_teste)

    # resultados
    print("\nResultados Algoritmo Machine Learning Based")

    qtd_comentarios_antes_mlb = pd.Series(y_teste).value_counts().sort_index()
    qtd_comentarios_depois_mlb = pd.Series(modelo_preds).value_counts().sort_index()
    acuracia_mlb = accuracy_score(y_teste, modelo_preds)
    f1_score_mlb = f1_score(y_teste, modelo_preds, average='weighted')
    matriz_mlb = confusion_matrix(y_teste, modelo_preds)

    print("\nQuantidade de comentarios por cada sentimento antes da classificacao - DATASET DE TESTE (dados desbalanceados):")
    print(qtd_comentarios_antes_mlb)

    print("\nQuantidade de comentarios por cada sentimento depois da classificacao - DATASET DE TESTE (dados desbalanceados):")
    print(qtd_comentarios_depois_mlb)

    print(f"\nAcuracia: {acuracia_mlb:.3f} ({acuracia_mlb * 100:.1f}%)")
    print(f"\nF1-Score: {f1_score_mlb:.3f}")

    print("\nMatriz de Confusao:")
    print(f"""
    Valores Negativos Corretos = {matriz_mlb[0][0]}
    Valores Neutros Corretos = {matriz_mlb[1][1]} 
    Valores Positivos Corretos = {matriz_mlb[2][2]}\n
    Valores Negativos classificados como Neutros = {matriz_mlb[0][1]}
    Valores Negativos classificados como Positivos = {matriz_mlb[0][2]}
    Valores Neutros classificados como Negativos = {matriz_mlb[1][0]}
    Valores Neutros classificados como Positivos = {matriz_mlb[1][2]}
    Valores Positivos classificados como Negativos = {matriz_mlb[2][0]}
    Valores Positivos classificados como Neutros = {matriz_mlb[2][1]}
    """)

    return {
        "qtd_comentarios_antes": qtd_comentarios_antes_mlb,
        "qtd_comentarios_depois": qtd_comentarios_depois_mlb,
        "acuracia": acuracia_mlb,
        "f1_score": f1_score_mlb
    }