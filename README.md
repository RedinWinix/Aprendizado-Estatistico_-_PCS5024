**Nome:**

# Atividade Banco de dados Adults
Para a tarefa de **PCS5024** foi instalado o banco de dados **Adults** onde foi feita  a análise das fetures e desenvolvimento de 3 classificadores, kNN, Regressão Logística e Redes Neurais Artificiais
## Análise das features
No banco de dados **Adults** temos as seguintes features:

| Nome da variável | Descrição                                                                                                             | Tipo                    |
| ---------------- | --------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| age              | Idade                                                                                                                 | Quantitativa contínua   |
| workclass        | Classe trabalhadora (Ex.: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov)                            | Qualitativa nominal     |
| fnlwgt           | Peso final                                                                                                            | Quantitativa contínua   |
| education        | Nível de escolaridade (Ex.: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th) | Qualitativa nominal     |
| education-num    | Número de anos de educação                                                                                            | Quantitativa contínua   |
| marital-status   | Estado civil (Ex.: Married-civ-spouse, Divorced, Never-married, Separated, Widowed)                                   | Qualitativa nominal     |
| occupation       | Profissão (Ex.: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners) | Qualitativa nominal     |
| relationship     | Parentesco (Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried)                                       | Qualitativa nominal     |
| race             | Cor/Raça (White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black)                                                | Qualitativa nominal     |
| sex              | Sexo (Feminino, Masculino)                                                                                            | Qualitativa nominal     |
| capital-gain     | Ganho                                                                                                                 | Quantitativa contínua   |
| capital-loss     | Prejuízo                                                                                                              | Quantitativa contínua   |
| hours-per-week   | Número de horas de trabalho por semana                                                                                | Quantitativa contínua   |
| native-country   | País de origem (Ex.: United States, Cambodia, England, Puerto Rico, Canada, Germany)                                  | Qualitativa nominal     |
| **income**       | **Indicadora se o salário anual é maior do que US$ 50.000 (>50K, <=50K)**                                             | **Qualitativa nominal** |

