import logging
import numpy as np
import pandas as pd
import seaborn as sns

# configurar logging
logging.basicConfig(
    filename='data/logs/processamento_dados.log',
    level=logging.INFO,
    format= '%(asctime)s - %(levelname)s \n%(message)s',
)

# verificacao de alteracoes no dataset
def verificar_dataset(mensagem,dataset,identificacao):
    logging.info(f'{mensagem}\n')
    logging.info(f"""Verificacao do dataset {identificacao.upper()}
Quantidade de linhas: {dataset.shape[0]}
Quantidade de colunas: {dataset.shape[1]}\n""")

# tratamento para valores ausentes por coluna
def valores_ausentes(dataset, identificacao):
    logging.info(f"Verificacao de valores ausentes no dataset {identificacao.upper()}.\n")
    dataset = dataset.replace(['Nan', 'nan', 'NaN', 'not specified', ''], np.nan)
    if dataset.isna().any(axis=1).any():
        total = dataset.isna().sum().sum()
        logging.info(f'Removendo {total} valores ausentes.\n')
        dataset = dataset.dropna()
        logging.info(f"Valores ausentes removidos. Total de linhas apos remocao: {dataset.shape[0]}\n")
    else:
        logging.info("Nao ha valores ausentes.\n")
    return dataset          

# tratamento para linhas duplicadas
def duplicatas(dataset,colunas,identificacao):
    logging.info(f"Verificacao de linhas duplicadas do dataset {identificacao.upper()}.\n")
    if dataset.duplicated(subset=colunas).any():
        logging.info(f'Removendo {dataset.duplicated(subset=colunas).sum()} linhas duplicadas.\n')
        dataset = dataset.drop_duplicates(subset=colunas, keep='last')
        logging.info(f"Linhas duplicadas removidas. Total de linhas apos remocao: {dataset.shape[0]}.\n")
    else:
        logging.info("Nao ha duplicatas.\n")

    return dataset

# traducao de valores de colunas com valores booleanos
def traducao_valores_booleanos(dataset,coluna):
    dataset[coluna] = dataset[coluna].map({'positive': 1, 'negative': -1, 'neutral': 0}).fillna(0).astype('Int64')

    return dataset

# excluir uma coluna do dataset
def excluir_coluna(dataset,coluna,identificacao):
    dataset = dataset.drop(coluna, axis = 1)
    logging.info(f'Coluna {coluna} foi excluida do dataset {identificacao.upper()}.\n')

    return dataset

# adicao de valores nas barras e colunas dos graficos
def valor_barras(plot, casas_decimais):
    for valor in plot.containers:
        plot.bar_label(valor, fmt=f'%.{casas_decimais}f', label_type='edge')

# geracao de graficos de colunas com uma variavel
def grafico_colunas_1(relacao, paleta):
    grafico = sns.barplot(x = relacao.index, y = relacao.values, hue = relacao.index, palette=paleta)
    return grafico

# geracao de grafico de barras com duas variaveis
def grafico_barras(relacao, coluna_x, tipo_relacao, hue, paleta):
    grafico = sns.barplot(data = relacao, x = tipo_relacao, y = coluna_x, hue = hue, palette=paleta)
    return grafico