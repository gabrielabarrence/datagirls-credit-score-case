# 💳 Credit Score Classification

> Projeto de Machine Learning para classificação de score de crédito utilizando análise exploratória, feature engineering, comparação entre modelos e explicabilidade com SHAP.

---

## 📖 Sobre o projeto

A Data Girls Finance quer tornar sua análise de crédito mais escalável, consistente e orientada por dados. O desafio foi transformar um histórico rico de clientes, com informações financeiras, comportamento de pagamento, endividamento e perfil de crédito, em um sistema capaz de classificar o `Credit_Score` em três faixas: `Good`, `Standard` e `Poor`.

O projeto seguiu uma lógica de negócio bastante realista. Primeiro, foi necessário entender a anatomia dos dados e corrigir problemas de qualidade que inviabilizariam qualquer modelo confiável: tipos incorretos, nulos explícitos e escondidos, valores sujos e outliers improváveis. Em seguida, a análise exploratória mostrou como o portfólio está distribuído e quais sinais parecem estar mais ligados ao risco de crédito. Por fim, foram comparados dois modelos com perfis bem diferentes: um modelo linear e altamente interpretável, e um modelo baseado em árvores, mais flexível para capturar interações complexas.

O resultado central do trabalho é que o comportamento de crédito não é linear. Clientes com maior risco não são explicados por uma única variável isolada, mas pela combinação entre atraso, dívida, quantidade de consultas, utilização do crédito e tempo de histórico. Por isso, o LightGBM se destacou claramente como a melhor solução analítica do pipeline.

---

# 📚 Documentação completa do pipeline

## 1️⃣ Leitura e exploração inicial

Foram utilizadas `pandas`, `numpy`, `matplotlib` e `seaborn` para carregar, inspecionar e visualizar os dados. Essas bibliotecas são escolhas adequadas porque:

- `pandas` é a base mais usada para manipulação tabular em ciência de dados;
- `numpy` oferece operações numéricas vetorizadas e suporte eficiente a arrays;
- `matplotlib` e `seaborn` permitem construir visualizações exploratórias e comparativas com rapidez e bom controle.

A exploração inicial validou dimensões, tipos e presença de nulos, além de confirmar que o dataset de teste não possuía a coluna alvo, o que evita vazamento de informação no processo de previsão.

---

## 2️⃣ Limpeza e preparação dos dados

A etapa de preparação foi extensa e necessária. O notebook converteu colunas com tipos errados, como `Age`, `Outstanding_Debt`, `Annual_Income` e `Credit_History_Age`, removendo caracteres espúrios e transformando os valores em formatos numéricos utilizáveis (`astype()`, `pd.to_numeric()`).

Também foi criada uma representação numérica mais útil para o histórico de crédito, convertendo anos e meses em meses totais e anos fracionários (`str.extract()`). Isso melhora a leitura estatística e a capacidade dos modelos de usar a informação temporal do histórico.

Para outliers, o notebook tomou uma decisão orientada por negócio: não remover indiscriminadamente valores altos que podem ser reais em finanças, como renda muito elevada, mas aplicar winsorização via `clip()` em colunas claramente distorcidas, como `Num_Bank_Accounts`, `Num_Credit_Card` e `Age`. Essa foi uma boa escolha porque preserva o volume de dados e reduz a influência de registros absurdos sem apagar completamente a observação.

Nos valores ausentes, houve duas frentes:

- nulos explícitos, tratados com mediana para variáveis numéricas e moda para categóricas (`fillna()`);
- nulos escondidos e placeholders como `_______`, `_` e `__`, convertidos antes para `NaN` (`replace()`).

Essa estratégia é adequada porque a mediana é robusta a assimetria e outliers, enquanto a moda preserva a categoria mais frequente sem criar rótulos artificiais. Também verificamos ausência de duplicados com `duplicated().sum()`.

---

## 3️⃣ Análise exploratória

A EDA mostrou que a classe `Standard` domina a base, com `53,17%`, seguida de `Poor` com `29,00%` e `Good` com `17,83%`. Isso sugere um portfólio concentrado em clientes medianos, com uma minoria em situação realmente favorável.

Destacamos que a distribuição por ocupação é relativamente homogênea e que clientes com score `Good` tendem a ser levemente mais velhos. Também foi observado que a maior concentração de renda mensal está entre 20 e 40 anos, tipicamente entre 2 e 4 mil em média.

---

## 4️⃣ Modelagem preditiva

### Modelo 1 — LightGBM

O modelo principal foi `lgb.LGBMClassifier()`, treinado com validação cruzada estratificada em 5 folds (`StratifiedKFold()`). O alvo foi codificado com `LabelEncoder()`.

O LightGBM é uma excelente escolha para esse problema, visto que lida muito bem com relações não lineares, captura interações entre variáveis automaticamente e costuma apresentar ótimo equilíbrio entre performance, velocidade e capacidade de generalização.

### Modelo 2 — Regressão Logística

Como baseline interpretável, foi usada uma pipeline com `ColumnTransformer()`, `StandardScaler()`, `OrdinalEncoder()` e `LogisticRegression()`.

Essa escolha também foi correta como comparação metodológica, porque regressão logística é um clássico de scorecard de crédito e fornece uma referência forte de interpretabilidade. Porém, por ser linear, ela tende a perder desempenho quando o problema depende de combinações complexas entre variáveis.

---

## 5️⃣ Avaliação dos modelos

As métricas escolhidas foram coerentes com um problema multiclasse de crédito:

- `accuracy_score()`
- `f1_score(average='macro')`
- `roc_auc_score(multi_class='ovr')`
- Gini
- Matriz de confusão
- FNR por classe (`confusion_matrix()`)

O uso de FNR por classe foi especialmente importante, porque em crédito errar clientes `Poor` significa aprovar ou priorizar mal perfis com maior risco. Isso é mais grave do que simplesmente perder alguns pontos de acurácia.

---

## 6️⃣ Explicabilidade

A interpretabilidade foi trabalhada com `shap.TreeExplainer()`, usando explicabilidade global e local.

Na análise global, foram avaliados os impactos médios das features por classe em uma amostra de 2000 linhas. Na análise local, explicamos a decisão para um cliente específico. Essa combinação é uma boa escolha porque permite unir com a visão executiva do comportamento do modelo.

---

# 🔍 Principais achados e insights

## Quais variáveis podem ter maior relação com a capacidade de pagamento dos clientes?

As variáveis mais ligadas à capacidade de pagamento são: fidelidade do cliente, dívida a pagar, atraso em pagamentos, tipo de empréstimo e intensidade do endividamento mensal. A lógica de negócio é clara: histórico mais longo tende a sinalizar comportamento mais estável, enquanto dívida pendente alta, mais atrasos e mais consultas no crédito podem indicar estresse financeiro ou crédito sendo buscado com maior urgência.

---

## Qual perfil de cliente possui maior probabilidade de ser classificado com score "Poor"?

O perfil mais associado a `Poor` tende a combinar sinais de fragilidade financeira: maior dívida pendente, histórico de crédito mais curto, atrasos recorrentes, mais consultas de crédito e pior composição de crédito (`Credit_Mix`). Não parece ser uma questão de ocupação isolada, pois a EDA indicou distribuição relativamente homogênea por profissão. O risco emerge mais da combinação entre comportamento financeiro e histórico de crédito do que de atributos demográficos simples.

---

## Um modelo de Machine Learning consegue classificar o score de crédito com boa performance?

Sim.

### LightGBM

- **AUC-ROC:** 0,9295
- **Gini:** 0,8590
- **Acurácia:** 0,8113
- **F1-Macro:** 0,8020

### Regressão Logística

- **AUC-ROC:** 0,7686
- **Acurácia:** 0,5929
- **F1-Macro:** 0,4896

O ponto mais crítico é o `FNR (Poor)`: o LightGBM erra cerca de `18,76%` dos clientes ruins, enquanto a regressão logística erra `56,18%`. Isso torna a RL inadequada para tomada de decisão operacional de crédito.

---

## Como os resultados do modelo poderiam apoiar uma equipe de crédito?

Há diversas formas de apoiar a equipe de crédito, algumas delas são:

- priorizar análises manuais em clientes com maior probabilidade de `Poor`;
- acelerar aprovações em perfis com forte sinal de `Good`, que tal uma automação baseada em ML?;
- revisar limites de clientes `Standard` com possíveis sinais de inadimplência;
- aproximar o relacionamento com clientes para prevenção de inadimplência;
- direcionar ações de educação financeira e retenção para clientes com histórico curto, alta dívida e sinais de risco crescente.

A comparação entre treino e teste previsto também sugere um ponto de atenção: o modelo estimou declínio da classe `Standard` para `50,39%` e queda de `Good` para `19%`. Isso pode indicar uma carteira nova mais conservadora, mais mediana ou um cenário de menor qualidade relativa dos clientes avaliados, visto que o Good não cresceu tanto.

---

## Quais cuidados éticos e de negócio devem ser considerados antes de usar o modelo em produção?

Antes de ir para produção, alguns cuidados são indispensáveis:

- validar viés indireto. Mesmo sem usar explicitamente atributos sensíveis, variáveis comportamentais podem funcionar como proxies;
- monitoramento do cenário econômico global, já que comportamento de crédito muda com contexto econômico;
- analisar custo de erro por classe, principalmente falsos negativos em `Poor` e falsos positivos em `Good`.

---

# 💡 Recomendações práticas

1. Adotar o LightGBM como modelo principal para a esteira analítica, mantendo a regressão logística apenas como baseline interpretável.

2. Levar para produção um pipeline formal de limpeza e padronização, porque boa parte do valor do projeto veio do tratamento rigoroso dos dados.

3. Usar os insights de SHAP para construir políticas de ação, por exemplo: revisão de limite para clientes com dívidas crescendo e muitas consultas, ou incentivo comercial para clientes com bom histórico e fidelidade longa.

4. Implementar monitoramento contínuo de métricas, especialmente `FNR` da classe `Poor`, além de distribuição das classes previstas.

5. Incluir governança ética e trilha de auditoria nas decisões apoiadas pelo modelo.

---

# 🎯 Conclusão

O projeto mostrou que é possível construir um classificador de credit score com boa performance e valor prático para negócio. O LightGBM foi a melhor escolha porque capturou padrões não lineares e entregou desempenho muito superior ao baseline linear.

Muito mais do que prever classes, o trabalho gerou uma visão total dos clientes: quais sinais indicam maior risco, quais perfis merecem atenção e como um time de crédito pode usar esse tipo de solução para priorizar análises, reduzir perdas e apoiar decisões com mais consistência.
