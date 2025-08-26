# Análise de Sentimento com Dados de E-commerce
Comparação de algoritmos de análise de sentimento feitos com base em regras (*rule-based*) e com aprendizado de máquina (*machine learning based*).  
O dataset utilizado (*Product Review*) foi disponibilizado por Mansi Thummar no *Kaggle*.

**Definição de Algoritmo de Sentimento *Rule-Based***: Abordagem em que se utiliza um conjunto de regras ou lexicons pré-definidos para classificar textos como positivos, negativos ou neutros. Neste projeto, foi aplicada a biblioteca *Vader* como base dessas regras.  
  
**Definição de Algoritmo de Sentimento *Machine Learning Based***: Abordagem em que se utiliza um modelo supervisionado que aprende padrões de sentimento a partir de dados rotulados. Neste projeto, foi implementada a Regressão Logística com divisão de 75% dos dados para treinamento e 25% para teste.

## Estrutura do Repositório
- `data/raw`: Arquivos com dados brutos, ainda não transformados.
- `data/processed`: Arquivos com dados que já passaram pelas etapas de transformação e limpeza.
- `data/logs`: Arquivos logs para monitoramento do pipeline.
- `src/utils.py`: Funções auxiliares para tratamento e transformação dos dados.
- `src/etl/extract.py`: Etapa de extração dos dados a partir de um arquivo CSV.
- `src/etl/transform.py`: Etapa de transformação e limpeza dos dados extraídos.
- `src/etl/load.py`: Etapa de carregamento dos dados transformados para `data/processed`.
- `src/models/rule_based.py`: Algoritmo de análise de sentimento baseado em regras.
- `src/models/ml_based.py`: Algoritmo de análise de sentimento com aprendizado de máquina.
- `src/visualization/generate_plots.py`: Etapa de geração de gráficos para análise dos resultados.
- `src/main.py`: Módulo principal que orquestra a execução de todo o pipeline, aplicação dos algoritmos e criação de gráficos.
- `results/figures`: Arquivos com imagens de gráficos gerados a partir de `generate_plots.py`.
- `README.md`: Documento que descreve o projeto e suas funcionalidades.

## Funcionalidades
1. Pipeline ETL (extract/extração, transform/transformação, load/carregamento) completo aplicado a um dataset de cerca de 77.000 registros.
2. Reserva de 25% de todos os dados como um conjunto desbalanceado de teste para validar e comparar o desempenho de ambos os algoritmos de forma padronizada.
3. Balanceamento de peso das classes é utilizado apenas no conjunto de treinamento no algoritmo *machine learning based*.
4. Classificação dos dados por meio de algoritmos *rule-based* e *machine learning based*.
5. Exibição de métricas de desempenho para cada algoritmo diretamente no terminal.
6. Geração de gráficos comparativos a partir de algumas métricas obtidas.

## Conceitos de Programação
- **Variáveis**: Utilizadas para armazenar dados temporários, configurações e resultados intermediários.  
- **Estruturas de Controle**: Utilizadas em estruturas condicionais e de repetição para gerenciar fluxos de dados e lógica de negócio.
- **Estruturas de Dados**: Utilização de listas e matrizes para armazenar e indexar sobre registros e resultados.
- **Funções**: Utilizadas para modularizar o código e garantir reutilização.
- **Tratamento de Erros**: Feito por meio de de blocos `try/except` para registrar possíveis falhas que possam ocorrer durante a execução do programa.

## Tecnologias Utilizadas
- **Python**: Linguagem principal utilizada para o desenvolvimento do projeto.
- **Logging**: Módulo nativo do Python para registro e monitoramento de todas as etapas do processamento.
- **Pandas**: Biblioteca utilizada para leitura, transformação e limpeza dos dados.
- **NumPy**: Biblioteca utilizada para realizar operações vetoriais e matemáticas rápidas.
- **Vader**: Biblioteca utilizada para formar dicionário lexicon e regras para estimar o sentimento (positivo, negativo ou neutro) em textos. 
- **Scikit-learn**: Biblioteca utilizada para desenvolver o algoritmo de aprendizado de máquina e calcular estatísticas de resultados.
- **Matplotlib** e **Seaborn**: Bibliotecas para a construção, visualização e estilização dos gráficos.

### Possíveis Melhorias Futuras
- Expandir o lexicon de palavras para melhorar a abordagem *rule-based*.
- Empregar GridSearch ou RandomSearch para otimizar hiperparâmetros do modelo de *machine learning*.
- Testar outros algoritmos de aprendizado de máquina para obter métricas melhores.
- Capturar e adicionar métricas extras para enriquecer os gráficos comparativos.