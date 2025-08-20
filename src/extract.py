import pandas as pd
from utils import verificar_dataset

try:
    dt = pd.read_csv("data/raw/store_product_reviews_raw.csv", sep = ',')
    verificar_dataset("Dataset importado com sucesso.",dt,"Teste")

except Exception as e:
    print(f'Não foi possível extrair os dados.\nErro: {e}')