#Author https://medium.com/@jameschen_78678/predict-gender-with-voice-and-speech-data-347f437fc4da
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Читаем набор данных
mydata = pd.read_csv("C:/Users/Кирилл/Desktop/ds/voice.csv") # считываем данные с таблицы voice
# Предварительный просмотр
mydata.head() # вызывает первые 5 строк
print(mydata.shape)  #определяем какой формы массив,определяем число элементов вдоль каждой оси массива(3168,21)
# Построение диаграмм
male = mydata.loc[mydata['label'] == 'male'] #используем метки для доступа к данным
female = mydata.loc[mydata['label'] == 'female'] #используем метки для доступа к данным
fig, axes = plt.subplots(10, 2, figsize=(10, 20)) #построение 20 графиков nrows=10 ,ncols=2
ax = axes.ravel() # делает массив плоским,
for i in range(20):
    ax[i].hist(male.iloc[:, i], bins=20, color=mglearn.cm3(0), alpha=.5) #вычисляем гистограмму набора данных
    ax[i].hist(female.iloc[:, i], bins=20, color=mglearn.cm3(2), alpha=.5) #вычисляем гистограмму набора данных
    ax[i].set_title(list(male)[i]) # добавляем заголовок
    ax[i].set_yticks(()) # устанавливаются метки 

ax[0].set_xlabel("Feature magnitude") #подпись оси х
ax[0].set_ylabel("Frequency") #подпись оси y
ax[0].legend(["male", "female"], loc="best") # табличка с подписью, что относится к male, что к female
fig.tight_layout() # чтобы не было наложений 
# Подготавливаем данные для моделирования 
mydata.loc[:, 'label'][mydata['label'] == "male"] = 0 #используем метки для доступа к данным
mydata.loc[:, 'label'][mydata['label'] == "female"] = 1 #используем метки для доступа к данным
mydata_train, mydata_test = train_test_split(mydata, random_state=0, test_size=.2) #выделяем выборку на тренировочную и тестовую часть
scaler = StandardScaler() # стандартизируем, приводим к одному масштабу 
scaler.fit(mydata_train.iloc[:, 0:20]) # подгонка  модели к обучающим данным
X_train = scaler.transform(mydata_train.iloc[:, 0:20]) #применяет преобразование к определенному набору примеров
X_test = scaler.transform(mydata_test.iloc[:, 0:20]) #применяет преобразование к определенному набору примеров
y_train = list(mydata_train['label'].values)
y_test = list(mydata_test['label'].values)
# Обучаем случайны лес
forest = RandomForestClassifier(n_estimators=5, random_state=0).fit(X_train, y_train)
print("Random Forests")
print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))



# Строим график 
def plot_feature_importances_mydata(model):
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center') # построение горизонтальной гистограммы
    plt.yticks(np.arange(n_features), list(mydata)) # подпись значений
    plt.xlabel("Variable importance") # подпись оси
    plt.ylabel("Independent Variable") # подпись оси

plot_feature_importances_mydata(forest)


plt.show()