import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score

scaler = MinMaxScaler()
encoder = LabelEncoder()

#ustawienei danych
data_file = pd.read_csv( "C:\\Users\\Admin\\Desktop\\PMN-26L-119134\\Zadanie 1\\iris.data", header=None)     # !!! tutaj proszę ustawić swoją ścieżkę do iris.data Mi nie działało dlatego dałem bezpośrednią
data = data_file.dropna()
categ = data.iloc[:, 4]
inf = data.iloc[:, :4]

#trening i test dane
tra_inf, test_inf, tra_categ, test_categ = train_test_split(inf, categ, test_size=0.2, random_state=30)
tra_inf = scaler.fit_transform(tra_inf)
test_inf = scaler.transform(test_inf)
tra_categ = encoder.fit_transform(tra_categ)
test_categ = encoder.transform(test_categ)

model = KNeighborsClassifier(n_neighbors=15)
model.fit(tra_inf, tra_categ)

#info 
prediction = model.predict(test_inf)
accuracy = accuracy_score(test_categ, prediction)
precision = precision_score(test_categ, prediction, average='weighted')
recall = recall_score(test_categ, prediction, average='weighted')
f1 = f1_score(test_categ, prediction, average='weighted')

matrix = confusion_matrix(test_categ, prediction)
print(f'Dokładność: {accuracy}\nPrecyzja: {precision}\nCzułość: {recall}\nF1-Score: {f1}\nMacierz błędów:\n{matrix}')

#rysowanie
all_inf = scaler.transform(inf)
all_categ = encoder.transform(categ)
tsne = TSNE(n_components=2, random_state=46)
draw = tsne.fit_transform(all_inf)

see = model.predict(all_inf)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,7))
graph1 = ax1.scatter(draw[:, 0], draw[:, 1], c=all_categ, cmap='viridis', s=80, edgecolors='k')
ax1.set_title('Original Iris data')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')


graph2 = ax2.scatter(draw[:, 0], draw[:, 1], c=see, cmap='viridis', s=80, edgecolors='k')
ax2.set_title("Machine vision")
ax2.set_xlabel('X')
ax2.set_ylabel('Y')

fig.colorbar(graph1, ax=[ax1, ax2], label='Species')

plt.show()