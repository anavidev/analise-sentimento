# Análise de Sentimento com Dados de E-commerce

## Estrutura do Repositório
- `data/raw`: Arquivos com dados brutos, ainda não transformados.
- `data/processed`: Arquivos com dados que já passaram pelas etapas de transformação e limpeza.
- `data/logs`: Arquivos logs para monitoramento e acompanhamento do processo de transformação dos dados.
- `src/utils.py`: Funções auxiliares para tratamento e transformação dos dados.
- `src/extract.py`: Etapa de extração dos dados a partir de um arquivo CSV.
- `src/transform.py`: Etapa de transformação e limpeza dos dados extraídos.
- `src/load.py`: Etapa de carregamento dos dados transformados para `data/processed`.
- `src/rule-based.py`: Algoritmo de análise de sentimento baseado em regras.
- `src/ml-based.py`: Algoritmo de análise de sentimento com aprendizado de máquina.
- `README.md`: Documento que descreve o projeto e suas funcionalidades.

## Funcionalidades

## Conceitos de Programação
- **Variáveis**: Utilizadas para armazenar dados do dataset e configurações personalizadas.
- **Estruturas de Controle**: Uso de estruturas condicionais e de repetição para manipulação e verificação de dados nos DataFrames.
- **Funções**: Utilizadas para modularizar o código e garantir reutilização.
- **Tratamento de Erros**: O tratamento de erros é feito por meio de de blocos `try/except` para registrar possíveis falhas que possam ocorrer durante a extração dos dados.

## Tecnologias Utilizadas
- **Python**: Linguagem principal utilizada para o desenvolvimento do projeto.
- **Pandas**: Biblioteca utilizada para leitura, manipulação e limpeza dos dados.
- **NumPy**: Biblioteca utilizada para realizar operações vetoriais e matemáticas rápidas.
- **Vader**: Biblioteca utilizada para formar dicionário lexicon e regras para estimar o sentimento (positivo, negativo ou neutro) em textos. 
- **Scikit-learn**: Biblioteca utilizada para desenvolver o algoritmo de aprendizado de máquina e calcular estatísticas de resultados.
- **Matplotlib** e **Seaborn**: Bibliotecas para a construção, visualização e estilização dos gráficos.

### Possíveis Melhorias Futuras