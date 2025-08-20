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
dt = dt.rename(traducao_colunas, axis = 1)

# retirar linhas e colunas desnecessarias do dataset
dt = excluir_coluna(dt,'nome_produto',"Teste")
dt = excluir_coluna(dt,'preco_produto',"Teste")
dt = excluir_coluna(dt,'pontuacao',"Teste")
dt = excluir_coluna(dt,'avaliacao',"Teste")

colunas = dt.columns

# validacao e limpeza dos dados
dt = valores_ausentes(dt,"Teste")
dt = duplicatas(dt,colunas,"Teste")

traducao_valores_booleanos(dt,'sentimento')

dt.to_csv('data/processed/store_product_reviews_transformed.csv', index=False, sep=',')

# verificacao final de tratamento de dados
arquivo_csv = pd.read_csv('data/processed/store_product_reviews_transformed.csv', sep=',')
arquivo_csv = valores_ausentes(arquivo_csv,"Teste")
arquivo_csv = duplicatas(arquivo_csv,colunas,"Teste")
verificar_dataset("Verificacao final.",arquivo_csv,"Teste")

print("Transformação concluida.")