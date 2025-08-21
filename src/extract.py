import pandas as pd
from utils import verificar_dataset

# dataset para o rule-based
try:
    dt_rb = pd.read_csv("data/raw/store_product_reviews_raw.csv", sep = ',')
    verificar_dataset("Dataset importado com sucesso.",dt_rb,"Rule-Based")

except Exception as e:
    print(f'Não foi possível extrair os dados.\nErro: {e}')


# dataset para o ml-based
try:
    dt_mlb = pd.read_csv("data/raw/amazon_reviews_raw.csv", sep = ';')
    verificar_dataset("Dataset importado com sucesso.",dt_mlb,"ML-Based")

except Exception as e:
    print(f'Não foi possível extrair os dados.\nErro: {e}')    