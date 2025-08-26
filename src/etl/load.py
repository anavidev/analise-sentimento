def carregamento_dados(dt_rb, dt_mlb):

    try:
        dt_rb.to_csv('data/processed/store_product_reviews_transformed.csv', index=False, sep=';', encoding='utf-8')
        dt_mlb.to_csv('data/processed/store_product_reviews_ml_transformed.csv', index=False, sep=';', encoding='utf-8')

        print("Carregamento concluido.")
    except FileNotFoundError as e:
        print(f'Verifique se o arquivo está no local correto.\nErro: {e}')
        exit(1)
    except Exception as e:
        print(f'Não foi possível extrair os dados.\nErro: {e}')
        exit(1)