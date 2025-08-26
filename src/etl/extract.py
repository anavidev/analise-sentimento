from utils.utils import pd, verificar_dataset

# dataset para o rule-based
def extracao_dados():

    try:
        dt = pd.read_csv("data/raw/store_product_reviews_raw.csv", sep = ';', encoding='utf-8')
        verificar_dataset("Dataset importado com sucesso.",dt,"Store Product")
        print("Extracao concluida.")

        return dt
    except FileNotFoundError as e:
        print(f'Verifique se o arquivo está no local correto.\nErro: {e}')
        exit(1)
    except Exception as e:
        print(f'Não foi possível extrair os dados.\nErro: {e}')
        exit(1)       