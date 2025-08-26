from utils.utils import verificar_dataset, valores_ausentes, duplicatas, excluir_coluna, traducao_valores_booleanos

## limpeza, validacao e transformacao dos dados
def transformacao_dados(dt):

    try:
        # traducao de colunas
        traducao_colunas = {
            'product_name': 'nome_produto',
            'product_price': 'preco_produto',
            'Rate': 'pontuacao',
            'Review': 'avaliacao',
            'Summary': 'comentario',
            'Sentiment': 'sentimento'
        }

        # dataframe rule-based
        dt_rb = dt.rename(traducao_colunas, axis = 1)
        colunas = dt_rb.columns

        # validacao e limpeza dos dados
        dt_rb = valores_ausentes(dt_rb,"Rule-Based")
        dt_rb = duplicatas(dt_rb,colunas,"Rule-Based")

        # retirar linhas e colunas desnecessarias do dataset
        for col in ['nome_produto', 'preco_produto', 'pontuacao', 'avaliacao']:
            dt_rb = excluir_coluna(dt_rb, col, "Rule-Based")

        dt_rb = traducao_valores_booleanos(dt_rb,'sentimento')

        verificar_dataset("Verificacao final.",dt_rb,"Rule-Based")

        # dataframe ml-based
        dt_mlb = dt.copy()

        dt_mlb = dt_mlb.rename(traducao_colunas, axis = 1)
        dt_mlb = valores_ausentes(dt_mlb,"ML-Based")
        dt_mlb = duplicatas(dt_mlb, dt_mlb.columns,"ML-Based")

        for col in ['nome_produto', 'preco_produto', 'pontuacao', 'avaliacao']:
            dt_mlb = excluir_coluna(dt_mlb, col, "ML-Based")

        dt_mlb = traducao_valores_booleanos(dt_mlb,'sentimento')

        verificar_dataset("Verificacao final.",dt_mlb,"ML-Based")

        print("Transformacao concluida.")

        return dt_rb, dt_mlb
    except Exception as e:
        print(f'Não foi possível transformar os dados.\nErro: {e}')
        exit(1)