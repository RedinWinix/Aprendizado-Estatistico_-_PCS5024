import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier



pd.set_option('display.max_columns', None)
encoder = OneHotEncoder(sparse_output=False)

#Lendo os dados de teste e treino
data_set = pd.read_csv("/home/redin-ubuntu/Studies/USP/Aprendizado Estatistico/adult/train_data.csv",
            sep=r'\s*,\s*',
            engine='python',
            na_values="?")

#Agora que carregado o banco de dados, vou passar a analisar as features, primeiro irei excluir a coluna id que não tem necessidade
data_set.drop(columns = ["Id"], inplace = True)


#Agora vou ver os dados faltantes
print(data_set.shape)
print(data_set.isna().sum())
#Isso significa que das 32559 entradas 4262 poderiam seriam perdidas. Aprox 13%, entretanto inicialmente não parece um problema e seguiremos a análise assim
data_set_clean = data_set.dropna()
print(len(data_set_clean))
print(len(data_set))


#tentando fazer uma mapa de correlação entre as features e foi criada uma variavel binaria que representa 1 para >50k e 0 para <=50k
#Criando variavel binária
data_set_clean["income_binary"] = data_set_clean["income"].map({"<=50K":0, ">50K": 1})

plt.figure()
sns.heatmap(data_set_clean[["age", "fnlwgt", "education.num", "capital.gain",
               "capital.loss", "hours.per.week", "income_binary"]].corr(), annot = True);
plt.title("Matriz de Correlações")
plt.xticks(rotation=45, ha='right')

#nota-se que a variavel fnlwgt não possui correlação linear com a income_binary e nenhuma outra variavel, dessa forma não faz sentido tela no data_set
data_set_clean.drop(columns = ["fnlwgt"], inplace = True)

#Agora iremos visualizar os dados
sns.displot(data_set_clean, x = 'age', bins = 25, color = "red") #Histograma da idade
plt.title("Gráfico da Idade")
plt.xticks(rotation=45, ha='right')

plt.figure()
data_set_clean['workclass'].value_counts().plot.bar(color=["blue", "green", "yellow", "purple","red", "gray"]);
plt.title("Gráfico Workclass")
plt.xticks(rotation=45, ha='right')

#A coluna "education" e "education.num" dizem a mesma coisa
plt.figure()
data_set_clean['education.num'].value_counts().plot.bar(color=["blue", "green", "yellow", "purple","red", "gray"]);
plt.title("Gráfico Education.num")
plt.xticks(rotation=45, ha='right')

plt.figure()
data_set_clean['education'].value_counts().plot.bar(color=["blue", "green", "yellow", "purple","red", "gray"]);
plt.title("Gráfico da Educação")
data_set_clean.drop(columns = ["education"], inplace = True)
plt.xticks(rotation=45, ha='right')

plt.figure()
data_set_clean['marital.status'].value_counts().plot.bar(color=["blue", "green", "yellow", "purple","red", "gray"]);
plt.title("Gráfico do Estado Civil")
plt.xticks(rotation=45, ha='right')

plt.figure()
data_set_clean['occupation'].value_counts().plot.bar(color=["blue", "green", "yellow", "purple","red", "gray"]);
plt.title("Gráfico das Profissões")
plt.xticks(rotation=45, ha='right')

plt.figure()
data_set_clean['relationship'].value_counts().plot.bar(color=["blue", "green", "yellow", "purple","red", "gray"]);
plt.title("Gráfico do Parentesco")
plt.xticks(rotation=45, ha='right')

plt.figure()
data_set_clean['race'].value_counts().plot.bar(color=["blue", "green", "yellow", "purple","red", "gray"]);
plt.title("Gráfico das Raças")
plt.xticks(rotation=45, ha='right')

plt.figure()
data_set_clean['sex'].value_counts().plot.pie(colors=["skyblue", "pink"]);
plt.title("Gráfico do Sexo")
plt.xticks(rotation=45, ha='right')

plt.figure()
data_set_clean['native.country'].value_counts().plot.bar(color=["blue", "green", "yellow", "purple","red", "gray", "brown", "orange", "black", "pink"]);
plt.title("Gráfico de Países Nativos")
plt.xticks(rotation=45, ha='right')


#uma estrategia verificada foi a de aglomerar os paises por criterios geograficos e socioeconomicos. dessa forma ficaram as novas categorias:
data_set_clean.loc[data_set_clean["native.country"] == "Hong", "native.country"] = "China"
data_set_clean.loc[data_set_clean["native.country"] == "Taiwan", "native.country"] = "China"
data_set_clean.loc[data_set_clean["native.country"] == "Honduras", "native.country"] = "Central-America"
data_set_clean.loc[data_set_clean["native.country"] == "Nicaragua", "native.country"] = "Central-America"
data_set_clean.loc[data_set_clean["native.country"] == "Haiti", "native.country"] = "Central-America"
data_set_clean.loc[data_set_clean["native.country"] == "Guatemala", "native.country"] = "Central-America"
data_set_clean.loc[data_set_clean["native.country"] == "Dominican-Republic", "native.country"] = "Central-America"
data_set_clean.loc[data_set_clean["native.country"] == "Cuba", "native.country"] = "Central-America"
data_set_clean.loc[data_set_clean["native.country"] == "El-Salvador", "native.country"] = "Central-America"
data_set_clean.loc[data_set_clean["native.country"] == "Puerto-Rico", "native.country"] = "Central-America"
data_set_clean.loc[data_set_clean["native.country"] == "Trinadad&Tobago", "native.country"] = "South-America"
data_set_clean.loc[data_set_clean["native.country"] == "Ecuador", "native.country"] = "South-America"
data_set_clean.loc[data_set_clean["native.country"] == "Peru", "native.country"] = "South-America"
data_set_clean.loc[data_set_clean["native.country"] == "Columbia", "native.country"] = "South-America"
data_set_clean.loc[data_set_clean["native.country"] == "Laos", "native.country"] = "Southeast-Asia"
data_set_clean.loc[data_set_clean["native.country"] == "Thailand", "native.country"] = "Southeast-Asia"
data_set_clean.loc[data_set_clean["native.country"] == "Cambodia", "native.country"] = "Southeast-Asia"
data_set_clean.loc[data_set_clean["native.country"] == "Vietnam", "native.country"] = "Southeast-Asia"
data_set_clean.loc[data_set_clean["native.country"] == "Holand-Netherlands", "native.country"] = "Western-Europe"
data_set_clean.loc[data_set_clean["native.country"] == "Scotland", "native.country"] = "Western-Europe"
data_set_clean.loc[data_set_clean["native.country"] == "Ireland", "native.country"] = "Western-Europe"
data_set_clean.loc[data_set_clean["native.country"] == "England", "native.country"] = "Western-Europe"
data_set_clean.loc[data_set_clean["native.country"] == "France", "native.country"] = "Western-Europe"
data_set_clean.loc[data_set_clean["native.country"] == "Greece", "native.country"] = "Southern-Europe"
data_set_clean.loc[data_set_clean["native.country"] == "Portugal", "native.country"] = "Southern-Europe"
data_set_clean.loc[data_set_clean["native.country"] == "Italy", "native.country"] = "Southern-Europe"
data_set_clean.loc[data_set_clean["native.country"] == "Yugoslavia", "native.country"] = "Central-Europe"
data_set_clean.loc[data_set_clean["native.country"] == "Hungary", "native.country"] = "Central-Europe"
data_set_clean.loc[data_set_clean["native.country"] == "Poland", "native.country"] = "Central-Europe"
data_set_clean.loc[data_set_clean["native.country"] == "Outlying-US(Guam-USVI-etc)", "native.country"] = "Others"
data_set_clean.loc[data_set_clean["native.country"] == "Jamaica", "native.country"] = "Others"
data_set_clean.loc[data_set_clean["native.country"] == "South", "native.country"] = "Others"

#Transformando raça e sexo em binario
data_set_clean["sex"] = (data_set_clean["sex"] == "Male").astype(int)
print(data_set_clean["sex"].head())


plt.xticks(rotation=45, ha='right')
plt.show();

#Agora é necessário dividir o data_set em treino e test
x = data_set_clean.drop(columns=["income_binary"])
y = data_set_clean["income_binary"]
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(X_test.head)


#Agora vamos transformar os dados qualitativos em numeros de fato
features = ["workclass", "marital.status", "occupation", "relationship", "race", "native.country"]
encoded_train = encoder.fit_transform(X_train[features])
encoded_test = encoder.transform(X_test[features])
qualitatives_encoded_train_set = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(), index=X_train.index)
qualitatives_encoded_test_set = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(), index=X_test.index)

print(qualitatives_encoded_train_set.head())
print (len(qualitatives_encoded_train_set.columns))

#Agora irei retirar as features qualitativas pois já estão no outro dicionario e criar o Y_train onde ficará o Y
X_train.drop(columns = ["workclass", "marital.status", "occupation", "relationship", "race", "native.country","income"], inplace = True)
X_test.drop(columns = ["workclass", "marital.status", "occupation", "relationship", "race", "native.country", "income"], inplace = True)
Quantitative_columns = X_train.columns
print(Quantitative_columns)

#Salvando o index antes do scalling
train_index = X_train.index
test_index = X_test.index

#Agora é necessário normalizar os dados
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#O sklearn transforma em numpy array, transformando de volta em DataFrame
X_train = pd.DataFrame(X_train, columns=Quantitative_columns, index=train_index)
X_test = pd.DataFrame(X_test, columns=Quantitative_columns, index=test_index)

#Concatenandos todas as features para treino
X_train = pd.concat([X_train, qualitatives_encoded_train_set], axis=1)
X_test = pd.concat([X_test, qualitatives_encoded_test_set], axis=1)

#Agora é a etapa para criar o modelo do kNN
#Fazendo validação cruzada para ver qual a melhor quantidade de n
best_k = 0
best_F1_kNN = 0
for k in range(1,51):
    knn = KNeighborsClassifier(n_neighbors=k)
    
    scores = cross_val_score(knn, X_train, Y_train, cv=5, scoring='f1')
    

    print(f"K={k}, F1 mean={scores.mean():.3f}")
    if(best_F1_kNN <= scores.mean()):
        best_F1_kNN = scores.mean()
        best_k = k

print(f"k:{best_k} ad F1:{best_F1_kNN}")
knn = KNeighborsClassifier(n_neighbors=best_k)


knn.fit(X_train, Y_train)
Y_pred_kNN = knn.predict(X_test)

print(f"\n{best_k}NN Results:")
print("Accuracy:", accuracy_score(Y_test, Y_pred_kNN))
print("Precision:", precision_score(Y_test, Y_pred_kNN))
print("Recall:", recall_score(Y_test, Y_pred_kNN))
print("F1 Score:", f1_score(Y_test, Y_pred_kNN))


#Agora Regressão Logística
logreg = LogisticRegression(max_iter=1000)

F1_logreg = cross_val_score(logreg, X_train, Y_train, cv=5, scoring='f1')
print(f"F1_log_reg: {F1_logreg.mean()}")

logreg.fit(X_train, Y_train)
Y_pred_lr = logreg.predict(X_test)

print("\nLogistic Regression Result:")
print("Accuracy:", accuracy_score(Y_test, Y_pred_lr))
print("Precision:", precision_score(Y_test, Y_pred_lr))
print("Recall:", recall_score(Y_test, Y_pred_lr))
print("F1 Score:", f1_score(Y_test, Y_pred_lr),"\n")


#Agora ANN com Multi Layer Perceptron
ann = MLPClassifier(
    hidden_layer_sizes=(50,),   # 1 hidden layer
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)

score = cross_val_score(ann, X_train, Y_train, cv=5, scoring='f1')
print(f"F1_ANN: {F1_logreg.mean()}")
ann.fit(X_train, Y_train)

Y_pred_ann = ann.predict(X_test)

print("\nANN Results:")
print("Accuracy:", accuracy_score(Y_test, Y_pred_ann))
print("Precision:", precision_score(Y_test, Y_pred_ann))
print("Recall:", recall_score(Y_test, Y_pred_ann))
print("F1 Score:", f1_score(Y_test, Y_pred_ann))