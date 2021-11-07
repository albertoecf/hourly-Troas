
## Read https://towardsdatascience.com/improving-random-forest-in-python-part-1-893916666cd

#%%
from pandas_profiling import ProfileReport
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Skicit-learn para hacer train_test
from sklearn.model_selection import train_test_split

# Importamos modelo Random Forest
from sklearn.ensemble import RandomForestRegressor

# %%


def ArmarFechaHora(df1):
    '''Nos permite armar un DateTime usando la fecha y sumando las horas'''
    df2 = df1
    df2['date'] = pd.to_datetime(df2['TIM_DAY'])
    df2['fechahora'] = pd.to_datetime(
        df2['date']) + df2['TIM_HOUR'].astype('timedelta64[h]')
    return df2


def puja(df1):
    ''''''
    df2 = df1

    df2['PujaTotal'] = df2['TROAS']*df2['CLICKS']
    # ROAS REAL = CONVERSION_VALUE / COST
    df2['ROASREAL'] = df2['CONVERSION_VALUE']/df2['COST']
    # SpreadDia  =  (TARGET ROAS SETEADO x CLICKS) - (ROAS REAL X CLICKS)
    df2['SpreadDia'] = df2['TROAS'] * \
        df2['CLICKS'] - df2['ROASREAL'] * df2['CLICKS']
    return df2


def costoMañana(df1):
    df2 = df1.copy()
    df2['costoMañana'] = df2['COST'].shift(-1)
    return df2


def metricas(df1):
    df2 = df1.copy()
    df2['CTR'] = df2['CLICKS'].div(df2['IMPRESSIONS'])
    df2['CPC'] = df2['COST'].div(df2['CLICKS'])
    return df2

def valorAnterior(df1,dias): 
    """Defasa todos los valores de un df en los días indicados"""
    df2 = df1.copy()
    for i in df2.columns: 
        df2[i+'SemPasada'] = df2[i].shift(dias)
    return df2

    # df2['costoSemPasada'] = df2['COST'].shift(dias -1)
    # df2['impSemPasada'] = df2['IMPRESSIONS'].shift(6)
    # return df2


# %%
# IMPORT FILE & CREATE DATETIME
raw = pd.read_csv('hogar.csv')
raw['date'] = pd.to_datetime(raw['TIM_DAY'])
raw['fechahora'] = pd.to_datetime(
    raw['date']) + raw['TIM_HOUR'].astype('timedelta64[h]')

# %%
hora = ArmarFechaHora(raw)
# %%
pujado = puja(hora)
# %%
# dia  = diario(pujado)
dia = pujado.groupby('TIM_DAY').agg({'TROAS': 'mean', 'CLICKS': 'sum', 'IMPRESSIONS': 'sum', 'COST': 'sum',
                                     'CONVERSION_VALUE': 'sum', 'PujaTotal': 'sum', 'SpreadDia': 'sum', 'ROASREAL': 'mean'})


# %%
conMetricas = metricas(dia)

completo = valorAnterior(conMetricas,7)
#%%
prepro = costoMañana(completo)
#%%
prepro.head()

# %%
# Calculamos un promedio del costo con un rolling window para tener de referencia
# ¿Cuánto mejor es nuestro modelo que tirar el promedio?
prepro['average'] = np.array(prepro.sort_index(ascending=False)[
                           'COST'].rolling(7).mean().sort_index(ascending=True).copy())


# ¿ A partir de qué día hubo cambios de TROAS? 
print('arrancan cambios roas')
prepro[prepro['TROAS'].pct_change()>0].head()
#%%
# En este caso, a partir del 2021-08-02
prepro = prepro[(prepro.index>'2021-08-11')] # &(prepro.index<'2021-10-30')].dropna()

#%% 

print('Correlación entre pct change de los valores')
prepro.pct_change().corr()['costoMañana'].sort_values(ascending=False)
#%%
print('correlación entre los valores')
prepro.corr()['costoMañana'].sort_values(ascending=False)

#prepro = prepro.pct_change().dropna()


#%%

#%%
# Usamos One-Hot encoding para las variables categoricas (en este caso dia semana)
features = pd.get_dummies(prepro[prepro.index <'2021-11-01'].dropna())

# %%

# labels : lo que queremos predecir, en este caso es el costo de Search
labels = np.array(features['costoMañana'])
# Tenemos que sacar costo de los features
# usamos axis = 1 para dropear a nivel columna
features = features.drop('costoMañana', axis=1)
# Nos guardamos la lista de los features
feature_list = list(features.columns)
# Convertimos los features a np arrays
features = np.array(features)


# %%
# Hacemos el train test split con test_size = 0.25
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.25, random_state=42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# %%
# Las predicciones base son el costo promedio de los ultimos 14 días
baseline_preds = test_features[:, feature_list.index('average')]
# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)
print('Average baseline error: ', round(np.mean(baseline_errors), 2), 'ARS')
# %%

# Iniciamos el modelo con 1.000 estimadores (deicision trees)
rf = RandomForestRegressor(n_estimators=1000, random_state=42)
# Entrenamos el modelo con nuestros datos
rf.fit(train_features, train_labels)

# %%

# Usamos nuestro model rf para hacer forecast con nuestros valores test
predictions = rf.predict(test_features)
# Calculamos el error absoluto entre nuestras predicciones y los valores reales de test
errors = abs(predictions - test_labels)
# Visualizamos el mae (la media del error absoluto/ mean absolute error)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'ARS.')
# %%
# Calculamos el MAPE (la media del error porcentual / mean absolute percentage error)
mape = 100 * (errors / test_labels)
print('Mean Absolute Percent Error:', round(np.mean(mape), 2), '%.')
# Calculamos y visualizamos accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
# %%


# usamos .feature_importances_ sobre nuestro modelo rf
# Para entender cuál es la importancia de cada variable (en %)
importances = list(rf.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2))
                       for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(
    feature_importances, key=lambda x: x[1], reverse=True)
# Visualizamos la importancia de cada variable
[print('Variable: {:20} Importance: {}'.format(*pair))
 for pair in feature_importances]
# %%


# Import matplotlib for plotting and use magic command for Jupyter Notebooks

# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation='vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance')
plt.xlabel('Variable')
plt.title('Variable Importances')
# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
