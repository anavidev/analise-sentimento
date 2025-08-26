from etl.extract import extracao_dados
from etl.transform import transformacao_dados
from etl.load import carregamento_dados
from models.rule_based import analisar_rule_based
from models.ml_based import analisar_ml_based
from visualization.generate_plots import gerar_graficos

def main():
    
    try:
        print("Etapa 1: Extração de dados")
        dt = extracao_dados()

        print("\nEtapa 2: Transformação de dados")
        dt_rb, dt_mlb = transformacao_dados(dt)

        print("\nEtapa 3: Carregamento de dados")
        carregamento_dados(dt_rb, dt_mlb)

        print("\nEtapa 4: Análise Algoritmo Rule-Based")
        resultados_rb = analisar_rule_based()

        print("\nEtapa 5: Análise Algoritmo Machine Learning Based")
        resultados_mlb = analisar_ml_based()

        print("\nEtapa 6: Criação de Gráficos")
        gerar_graficos(resultados_rb, resultados_mlb)

    except Exception as e:
        print(f"O pipeline falhou durante a execução.\nErro: {e}")
        exit(1)

if __name__ == "__main__":   
    main()