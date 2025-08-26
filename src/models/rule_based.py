from sklearn.model_selection import train_test_split
from utils.utils import pd, verificar_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analisar_rule_based():
    analisador = SentimentIntensityAnalyzer()

    dt_rb = pd.read_csv("data/processed/store_product_reviews_transformed.csv", sep = ';')
    verificar_dataset("Dataset importado com sucesso.",dt_rb,"Rule-Based")

    print("Analise em andamento.")

    # separacao de 25% dos dados para teste e classificacao das pontuacoes
    dt_rb_treino, dt_rb_teste = train_test_split(
        dt_rb,
        test_size=0.25,
        random_state=42,
        stratify=dt_rb['sentimento'])

    dt_rb_teste['pontuacao_vader'] = dt_rb_teste['comentario'].apply(lambda texto: analisador.polarity_scores(texto))
    dt_rb_teste['compound_vader'] = dt_rb_teste['pontuacao_vader'].apply(lambda x: x['compound'])
    dt_rb_teste['sentimento_pred'] = dt_rb_teste['compound_vader'].apply(
        lambda x: 1 if x >= 0.05
        else -1 if x <= -0.05
        else 0
    )

    # resultados
    print("\nResultados Algoritmo Rule-Based\n")

    qtd_comentarios_antes_rb = dt_rb_teste.groupby('sentimento')['compound_vader'].count().reset_index(name='total_comentarios_antes_rb')
    qtd_comentarios_depois_rb = dt_rb_teste.groupby('sentimento_pred')['compound_vader'].count().reset_index(name='total_comentarios_depois_rb')
    acuracia_rb = accuracy_score(dt_rb_teste['sentimento'], dt_rb_teste['sentimento_pred'])
    matriz_rb = confusion_matrix(dt_rb_teste['sentimento'], dt_rb_teste['sentimento_pred'])
    f1_score_rb = f1_score(dt_rb_teste['sentimento'], dt_rb_teste['sentimento_pred'], average='weighted')

    print("Quantidade de comentarios por cada sentimento antes da classificacao:")
    print(qtd_comentarios_antes_rb)

    print("\nQuantidade de comentarios por cada sentimento depois da classificacao:")
    print(qtd_comentarios_depois_rb)

    print(f"\nAcuracia: {acuracia_rb:.3f} ({acuracia_rb * 100:.1f}%)")
    print(f"\nF1-Score: {f1_score_rb:.3f}")

    print("\nMatriz de Confusao:")
    print(f"""
    Valores Negativos Corretos = {matriz_rb[0][0]}
    Valores Neutros Corretos = {matriz_rb[1][1]} 
    Valores Positivos Corretos = {matriz_rb[2][2]}\n
    Valores Negativos classificados como Neutros = {matriz_rb[0][1]}
    Valores Negativos classificados como Positivos = {matriz_rb[0][2]}
    Valores Neutros classificados como Negativos = {matriz_rb[1][0]}
    Valores Neutros classificados como Positivos = {matriz_rb[1][2]}
    Valores Positivos classificados como Negativos = {matriz_rb[2][0]}
    Valores Positivos classificados como Neutros = {matriz_rb[2][1]}
    """)

    return {
        "qtd_comentarios_antes": qtd_comentarios_antes_rb,
        "qtd_comentarios_depois": qtd_comentarios_depois_rb,
        "acuracia": acuracia_rb,
        "f1_score": f1_score_rb
    }