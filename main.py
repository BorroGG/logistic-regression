import numpy as np  # работа с матрицами
import pandas as pd  # для работы с датасетами
from sklearn.model_selection import train_test_split  # для разбивки данных на тестовые и обучающие
from sklearn.preprocessing import StandardScaler  # для нормализации данных
from sklearn.metrics import accuracy_score  # для проверки точности


# логистическая функция, сигмоид
def calc_sigmoid(z):
    p = 1 / (1 + np.exp(-z))
    p = np.minimum(p, 0.9999)
    p = np.maximum(p, 0.0001)
    return p


# вычисление логисчтической функции
def calc_pred_func(theta, x):
    y = np.dot(theta, np.transpose(x))  # np.dot - скалярное произведение, np.transpose(x) транспонирование матрицы
    return calc_sigmoid(y)


# вычисление ошибки
# максимальное правдоподобие
def calc_error(y_pred, y_label):
    cost = (-y_label * np.log(y_pred) - (1 - y_label) * np.log(1 - y_pred)).mean()
    return cost


# градиентный спуск
def gradient_descent(y_pred, y_label, x, learning_rate, theta):
    len_label = len(y_label)
    J = (-(np.dot(np.transpose(x), (y_label - y_pred))) / len_label)  # формула градиента
    theta -= learning_rate * J  # learning_rate - скорость обучения, шаг спуска, theta веса
    # print("theta_0 shape: ",np.shape(theta),np.shape(J))
    return theta


# расчет евклидова расстояния sqrt((point_1 - point_2)^2 + (point_3 - point_4)^2)
def distance(y_pred_old, y_pred):
    square = np.square(y_pred_old - y_pred)
    sum_square = np.sum(square)
    distances = np.sqrt(sum_square)
    return distances


# обучение модели
def train_while(y_label, x, learning_rate, theta):
    count_iter = 0
    check_distance = 100
    while check_distance >= 0.001:
        y_pred = calc_pred_func(theta, x)
        theta_old = np.array(theta)
        theta = gradient_descent(y_pred, y_label, x, learning_rate, theta)
        count_iter += 1
        check_distance = distance(theta_old, theta)
    print("count iteration in while:", count_iter)
    return theta


# обучение модели
def train_iter(y_label, x, learning_rate, theta, iter):
    for i in range(iter):
        y_pred = calc_pred_func(theta, x)
        theta = gradient_descent(y_pred, y_label, x, learning_rate, theta)
    return theta


# вычисление вероятности классов
def predict(x_test, theta):
    y_test = np.dot(theta, np.transpose(x_test))

    p = calc_sigmoid(y_test)
    np.set_printoptions(suppress=True)  # для читабельности значений в матрице
    # print(p)

    p = p >= 0.5  # порог 0.5
    y_pred = np.zeros(p.shape[0])
    for i in range(len(y_pred)):
        if p[i]:
            y_pred[i] = 1
        else:
            continue

    return y_pred


# ЧАСТЬ 1
# Импорт таблицы с данными
diabetes = pd.read_csv('diabetes.csv')

# Проверка данных
diabetes3 = diabetes.loc[diabetes['Диагноз'] == 1]
counterFunc = diabetes.apply(
    lambda x: True if x[8] == 1 else False, axis=1)
numOfRows = len(counterFunc[counterFunc == False].index)
# Узнаем, что дигнозы распределенны не равномерно, 268 = 1 and 500 = 0
# print("diabetes3 = \n", diabetes3)
# print("numOfRows = ", numOfRows)

# Дублируем данные меньшего объема для нормализации
diabetes = diabetes.append(diabetes3, ignore_index=True)
# print("diabetes = \n", diabetes)

col = diabetes.columns

# Разбиене данных на признаки и ответ
x_labels = col[:-1]
y_label = col[-1:]

x, y = diabetes.loc[:, x_labels], diabetes.loc[:, y_label]  # .loc доступ к группе строк и столбцов
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)
# разделение данных на обучающую и тестовую в формате 70/30, random_state число для псевдорандома

# нормализация признаковых данных
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# обучение модели, с шагом в 0.000_001, все стартовые веса взяты за 1, количество шагов 900
theta = train_iter(np.reshape(y_train.shape[0], 1), x_train, 0.00001, np.array([1.] * 8), 900)

# проверка на тестовых данных
y_pred = predict(x_test, theta)

# оценка точности модели
print('accuracy_score with iter = ', accuracy_score(y_test, y_pred))

# обучение модели, с шагом в 0.000_001, все стартовые веса взяты за 1
theta = train_while(np.reshape(y_train.shape[0], 1), x_train, 0.00001, np.array([1.] * 8))

# проверка на тестовых данных
y_pred = predict(x_test, theta)

# оценка точности модели
print('accuracy_score with while = ', accuracy_score(y_test, y_pred))

# Часть 2
# Карта характеристик корреляций
corr = diabetes.corr()
# print(corr)

# Корреляция с диагнозом
cor_target = abs(corr["Диагноз"])
# Выбрать наименее корреллирующие значения
relevant_features = cor_target[cor_target < 0.1]
print(relevant_features)

# Удаление наименее коррелирующих столбцов
diabetes.drop('АД', inplace=True, axis=1)
diabetes.drop('Толщина КС', inplace=True, axis=1)
col = diabetes.columns
# print('Таблица после удаления столбцов: \n', diabetes)

# Повторение обучения и тестирования модели
x_labels = col[:-1]
y_label = col[-1:]

x, y = diabetes.loc[:, x_labels], diabetes.loc[:, y_label]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

theta = train_iter(np.reshape(y_train.shape[0], 1), x_train, 0.00001, np.array([1.] * 6), 900)

y_pred = predict(x_test, theta)

print('accuracy_score with iter = ', accuracy_score(y_test, y_pred))

theta = train_while(np.reshape(y_train.shape[0], 1), x_train, 0.00001, np.array([1.] * 6))

y_pred = predict(x_test, theta)

print('accuracy_score with while = ', accuracy_score(y_test, y_pred))
# До удаления коррелирующих данных    - accuracy_score =  0.6796536796536796
# После удаления коррелирующих данных - accuracy_score =  0.7142857142857143
