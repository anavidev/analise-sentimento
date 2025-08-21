from transform import *

arquivo_csv_rb.to_csv('data/processed/store_product_reviews_transformed.csv', index=False, sep=',')
arquivo_csv_mlb.to_csv('data/processed/amazon_reviews_transformed.csv', index=False, sep=';')