import logging

# configuracao do logging
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
def valores_ausentes(dataset,identificacao):
    logging.info(f"Verificacao de valores ausentes do dataset {identificacao.upper()}.\n")
    if dataset.isna().any(axis=1).any():
        logging.info(f'Removendo {dataset.isna().sum()} valores ausentes.\n')
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
        dataset = dataset.drop_duplicates(subset=colunas, keep='last') # mantem a ultima ocorrencia
        logging.info(f"Linhas duplicadas removidas. Total de linhas apos remocao: {dataset.shape[0]}.\n")
    else:
        logging.info("Nao ha duplicatas.\n")

    return dataset


# traducao de valores de colunas com valores booleanos
def traducao_valores_booleanos(dataset,coluna):
    dataset[coluna] = dataset.apply(
    lambda row: 1 if row[coluna] == 'positive'
    else -1 if row[coluna] == 'negative'
    else 0,
    axis = 1
    )

    return dataset


# excluir uma coluna do dataset
def excluir_coluna(dataset,coluna,identificacao):
    dataset = dataset.drop(coluna, axis = 1)
    logging.info(f'Coluna {coluna} foi excluida do dataset {identificacao.upper()}.\n')

    return dataset


# pesos das novas categorias
def classificar_sentimento_novo(pontuacao):
    if pontuacao >= 1:
        return 1
    elif pontuacao <= -1:
        return -1
    else:
        return 0


# aplicar pesos nas categorias do algoritmo baseado em regras
def peso_categoria_positiva(dataset, categoria, peso):
    for palavra in categoria:
        dataset["pontuacao"] += peso * dataset["comentario"].str.casefold().str.count(palavra)

def peso_categoria_negativa(dataset, categoria, peso):
    for palavra in categoria:
        dataset["pontuacao"] -= peso * dataset["comentario"].str.casefold().str.count(palavra)