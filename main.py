from keras import models
from keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def import_data():
    df_bp = []
    df_bp = pd.read_csv('x_data/X_bp.csv')
    df_nup = []
    df_nup = pd.read_csv('x_data/X_nup.csv')
    df = df_bp.merge(df_nup, left_on='Unnamed: 0', right_on='Unnamed: 0')
    return df


data = import_data()
data.drop(['Unnamed: 0'], inplace=True, axis=1)

for x in data:
    q75, q25 = np.percentile(data.loc[:, x], [75, 25])
    intr_qr = q75 - q25
    max = q75 + (1.5 * intr_qr)
    min = q25 - (1.5 * intr_qr)
    data.loc[data[x] < min, x] = np.nan
    data.loc[data[x] > max, x] = np.nan

data.head()
data = data.dropna(axis = 0)


def get_dfpm(dfp):
    # transformer = MinMaxScaler().fit(dfp)
    scaler = MinMaxScaler()
    col = dfp.columns
    processed = scaler.fit_transform(dfp)
    dfpm = pd.DataFrame(processed, columns=col)
    return dfpm


dfpm = get_dfpm(data)


def build_model(xtrn):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(xtrn.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


result = ['Соотношение матрица-наполнитель']
inputcol = ['Плотность, кг/м3', 'модуль упругости, ГПа',
            'Количество отвердителя, м.%', 'Содержание эпоксидных групп,%_2',
            'Температура вспышки, С_2', 'Поверхностная плотность, г/м2',
            'Модуль упругости при растяжении, ГПа', 'Прочность при растяжении, МПа',
            'Потребление смолы, г/м2', 'Угол нашивки, град',
            'Шаг нашивки', 'Плотность нашивки']
x_train = dfpm[inputcol]
y_train = dfpm[result]
xtrn, xtest, ytrn, ytest = train_test_split(x_train, y_train, test_size=0.3)
model = build_model(xtrn)

k = 4
num_val_samples = len(xtrn) // k
num_epochs = 50
all_scores = []
for i in range(k):
    val_data = xtrn[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = ytrn[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [xtrn[:i * num_val_samples],
         xtrn[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [ytrn[:i * num_val_samples],
         ytrn[(i + 1) * num_val_samples:]],
        axis=0)

model.fit(
    partial_train_data,
    partial_train_targets,
    epochs=num_epochs,
    batch_size=4,
    validation_data=(val_data, val_targets),
    verbose=0
)
transformer = MinMaxScaler().fit(data)
