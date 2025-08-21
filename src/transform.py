from extract import *
from utils import *

## limpeza, validacao e transformacao dos dados

# dataframe de teste

# traducao de colunas
traducao_colunas = {
    'product_name': 'nome_produto',
    'product_price': 'preco_produto',
    'Rate': 'pontuacao',
    'Review': 'avaliacao',
    'Summary': 'comentario',
    'Sentiment': 'sentimento'
}
dt_rb = dt_rb.rename(traducao_colunas, axis = 1)

# retirar linhas e colunas desnecessarias do dataset
dt_rb = excluir_coluna(dt_rb,'nome_produto',"Rule-Based")
dt_rb = excluir_coluna(dt_rb,'preco_produto',"Rule-Based")
dt_rb = excluir_coluna(dt_rb,'pontuacao',"Rule-Based")
dt_rb = excluir_coluna(dt_rb,'avaliacao',"Rule-Based")

colunas = dt_rb.columns

# validacao e limpeza dos dados
dt_rb = valores_ausentes(dt_rb,"Rule-Based")
dt_rb = duplicatas(dt_rb,colunas,"Rule-Based")

traducao_valores_booleanos(dt_rb,'sentimento')

dt_rb.to_csv('data/processed/store_product_reviews_transformed.csv', index=False, sep=',')

# verificacao final de tratamento de dados
arquivo_csv_rb = pd.read_csv('data/processed/store_product_reviews_transformed.csv', sep=',')
arquivo_csv_rb = valores_ausentes(arquivo_csv_rb,"Rule-Based")
arquivo_csv_rb = duplicatas(arquivo_csv_rb,colunas,"Rule-Based")
verificar_dataset("Verificacao final.",arquivo_csv_rb,"Rule-Based")


# dataframe de treinamento

traducao_colunas = {
    'Text': 'comentario',
    'cleaned_review': 'comentario_limpo',
    'Score': 'avaliacao',
    'sentiment_score': 'pontuacao',
    'sentiment_category': 'sentimento',
    'fear': 'medo',
    'anger': 'raiva',
    'anticip': 'antecipacao',
    'trust': 'confianca',
    'surprise': 'surpresa',
    'positive': 'positividade',
    'negative': 'negatividade',
    'sadness': 'tristeza',
    'disgust': 'nojo',
    'joy': 'alegria',
    'topic': 'topico'
}
dt_mlb = dt_mlb.rename(traducao_colunas, axis = 1)


dt_mlb = excluir_coluna(dt_mlb,'comentario_limpo',"ML-Based")
dt_mlb = excluir_coluna(dt_mlb,'avaliacao',"ML-Based")
dt_mlb = excluir_coluna(dt_mlb,'pontuacao',"ML-Based")
dt_mlb = excluir_coluna(dt_mlb,'medo',"ML-Based")
dt_mlb = excluir_coluna(dt_mlb,'raiva',"ML-Based")
dt_mlb = excluir_coluna(dt_mlb,'antecipacao',"ML-Based")
dt_mlb = excluir_coluna(dt_mlb,'confianca',"ML-Based")
dt_mlb = excluir_coluna(dt_mlb,'surpresa',"ML-Based")
dt_mlb = excluir_coluna(dt_mlb,'positividade',"ML-Based")
dt_mlb = excluir_coluna(dt_mlb,'negatividade',"ML-Based")
dt_mlb = excluir_coluna(dt_mlb,'tristeza',"ML-Based")
dt_mlb = excluir_coluna(dt_mlb,'nojo',"ML-Based")
dt_mlb = excluir_coluna(dt_mlb,'alegria',"ML-Based")
dt_mlb = excluir_coluna(dt_mlb,'topico',"ML-Based")
dt_mlb = excluir_coluna(dt_mlb,'cluster',"ML-Based")
dt_mlb = excluir_coluna(dt_mlb,'pca_1',"ML-Based")
dt_mlb = excluir_coluna(dt_mlb,'pca_2',"ML-Based")

colunas = dt_mlb.columns


dt_mlb = valores_ausentes(dt_mlb,"ML-Based")
dt_mlb = duplicatas(dt_mlb,colunas,"ML-Based")

traducao_valores_booleanos(dt_mlb,'sentimento')

dt_mlb.to_csv('data/processed/amazon_reviews_transformed.csv', index=False, sep=';')


arquivo_csv_mlb = pd.read_csv('data/processed/amazon_reviews_transformed.csv', sep=';')
arquivo_csv_mlb = valores_ausentes(arquivo_csv_mlb,"ML-Based")
arquivo_csv_mlb = duplicatas(arquivo_csv_mlb,colunas,"ML-Based")
verificar_dataset("Verificacao final.",arquivo_csv_mlb,"ML-Based")

print("Transformação concluida.")