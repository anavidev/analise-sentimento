from transform import *

dt_rb = arquivo_csv.to_csv('data/processed/store_product_reviews_transformed.csv', index=False, sep=';')
dt_mlb = arquivo_csv.to_csv('data/processed/store_product_reviews_ml_transformed.csv', index=False, sep=';')

print("Carregamento concluido.")