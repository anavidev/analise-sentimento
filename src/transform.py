from extract import *
from utils import valores_ausentes, duplicatas, excluir_coluna, traducao_valores_booleanos, balancear_classes

## limpeza, validacao e transformacao dos dados

# traducao de colunas
traducao_colunas = {
    'product_name': 'nome_produto',
    'product_price': 'preco_produto',
    'Rate': 'pontuacao',
    'Review': 'avaliacao',
    'Summary': 'comentario',
    'Sentiment': 'sentimento'
}
dt = dt.rename(traducao_colunas, axis = 1)
colunas = dt.columns

# validacao e limpeza dos dados
dt = valores_ausentes(dt,"Store Product")
dt = duplicatas(dt,colunas,"Store Product")

# retirar linhas e colunas desnecessarias do dataset
dt = excluir_coluna(dt,'nome_produto',"Store Product")
dt = excluir_coluna(dt,'preco_produto',"Store Product")
dt = excluir_coluna(dt,'pontuacao',"Store Product")
dt = excluir_coluna(dt,'avaliacao',"Store Product")
colunas = dt.columns

dt = traducao_valores_booleanos(dt,'sentimento')
dt = balancear_classes(dt,'sentimento',"Store Product")

dt.to_csv('data/processed/store_product_reviews_transformed.csv', index=False, sep=';')

# verificacao final de tratamento de dados
arquivo_csv_rb = pd.read_csv('data/processed/store_product_reviews_transformed.csv', sep=';') # dataframe rule-based

arquivo_csv_mlb = arquivo_csv_rb.copy()
arquivo_csv_mlb.to_csv('data/processed/store_product_reviews_ml_transformed.csv', index=False, sep=';') # dataframe ml-based

verificar_dataset("Verificacao final - Dataframe Rule-Based.",arquivo_csv_rb,"Store Product Rule-Based")
verificar_dataset("Verificacao final - Dataframe Machine Learning Based",arquivo_csv_mlb,"Store Product ML-Based")

print("Transformacao concluida.")