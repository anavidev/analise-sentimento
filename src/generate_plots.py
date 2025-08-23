import matplotlib.pyplot as plt
from utils import pd, valor_barras, grafico_colunas_1, grafico_barras
from rule_based import qtd_comentarios_antes_rb, qtd_comentarios_depois_rb, acuracia_rb, f1_score_rb
from ml_based import qtd_comentarios_antes_mlb, qtd_comentarios_depois_mlb, acuracia_mlb, f1_score_mlb


# comparacao de acuracia e f1-score entre rule-based e ml-based
dados_acuracia = pd.DataFrame({
    'Algoritmo': ['Rule-Based', 'Machine Learning'],
    'Acurácia': [acuracia_rb * 100, acuracia_mlb * 100]
})

f1_data = pd.DataFrame({
    'Algoritmo': ['Rule-Based', 'Machine Learning'],
    'F1_Score': [f1_score_rb, f1_score_mlb]
})

plt.figure(figsize=(10, 5))

plt.subplot(1,2,1)
plot_acuracia = grafico_colunas_1(dados_acuracia.set_index('Algoritmo')['Acurácia'], 'colorblind')
valor_barras(plot_acuracia, 1)
plot_acuracia.set_ylim(top=100)
plt.title('Comparação de Acurácia')
plt.xlabel('Algoritmos')
plt.ylabel('Acurácia (%)')

plt.subplot(1,2,2)
plot_f1 = grafico_colunas_1(f1_data.set_index('Algoritmo')['F1_Score'], 'colorblind')
valor_barras(plot_f1, 3)
plot_f1.set_ylim(top=1)
plt.title('Comparação de F1-Score')
plt.xlabel('Algoritmos')
plt.ylabel('F1-Score')

plt.tight_layout()
# plt.show()

plt.savefig("results/figures/comparacao_acuracia.png", dpi=300, bbox_inches='tight')

# distribuicao de sentimentos do rule-based e ml-based

rb_antes = qtd_comentarios_antes_rb.rename(columns={'sentimento': 'Sentimento', 'total_comentarios_antes_rb': 'Quantidade'})
rb_depois = qtd_comentarios_depois_rb.rename(columns={'sentimento_pred': 'Sentimento', 'total_comentarios_depois_rb': 'Quantidade'})

rb_antes['Sentimento'] = rb_antes['Sentimento'].map({-1: 'Negativo', 0: 'Neutro', 1: 'Positivo'})
rb_depois['Sentimento'] = rb_depois['Sentimento'].map({-1: 'Negativo', 0: 'Neutro', 1: 'Positivo'})

rb_antes['Momento'] = 'Antes'
rb_depois['Momento'] = 'Depois'
rb_completo = pd.concat([rb_antes, rb_depois])


mlb_antes = qtd_comentarios_antes_mlb.reset_index()
mlb_antes.columns = ['Sentimento', 'Quantidade']
mlb_depois = qtd_comentarios_depois_mlb.reset_index()
mlb_depois.columns = ['Sentimento', 'Quantidade']
mlb_antes['Sentimento'] = mlb_antes['Sentimento'].map({-1: 'Negativo', 0: 'Neutro', 1: 'Positivo'})
mlb_depois['Sentimento'] = mlb_depois['Sentimento'].map({-1: 'Negativo', 0: 'Neutro', 1: 'Positivo'})

mlb_antes['Momento'] = 'Antes'
mlb_depois['Momento'] = 'Depois'
mlb_completo = pd.concat([mlb_antes, mlb_depois])

plt.figure(figsize=(15, 5))

plt.subplot(1,2,1)
sentimentos_rb = grafico_barras(rb_completo, 'Sentimento', 'Quantidade', 'Momento', 'colorblind')
valor_barras(sentimentos_rb, 0)
sentimentos_rb.set_xlim(right=7000)
plt.title('Distribuição de Sentimentos - Rule-Based (Antes e Depois)')
plt.xlabel('Quantidade de Comentários')
plt.ylabel('Sentimento')
plt.legend(title='Momento')

plt.subplot(1,2,2)
sentimentos_mlb = grafico_barras(mlb_completo, 'Sentimento', 'Quantidade', 'Momento', 'colorblind')
valor_barras(sentimentos_mlb, 0)
sentimentos_mlb.set_xlim(right=2000)
plt.title('Distribuição de Sentimentos - Machine Learning (Antes e Depois)')
plt.xlabel('Quantidade de Comentários')
plt.ylabel('Sentimento')
plt.legend(title='Momento')

plt.tight_layout()
# plt.show()

plt.savefig("results/figures/distribuicao_sentimentos.png", dpi=300, bbox_inches='tight')