from transform import arquivo_csv_rb, arquivo_csv_mlb

dt_rb = arquivo_csv_rb.to_csv('data/processed/store_product_reviews_transformed.csv', index=False, sep=';')
dt_mlb = arquivo_csv_mlb.to_csv('data/processed/store_product_reviews_ml_transformed.csv', index=False, sep=';')

print("Carregamento concluido.")