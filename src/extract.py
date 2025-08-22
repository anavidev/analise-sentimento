import pandas as pd
from utils import verificar_dataset

# dataset para o rule-based
try:
    dt = pd.read_csv("data/raw/store_product_reviews_raw.csv", sep = ';')
    verificar_dataset("Dataset importado com sucesso.",dt,"Store Product")

except Exception as e:
    print(f'Não foi possível extrair os dados.\nErro: {e}')
