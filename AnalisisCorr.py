# %%
import pandas as pd
from pandas.core.indexes import period
import seaborn as sns
import numpy as np
import matplotlib.pyplot  as plt
# Skicit-learn para hacer train_test
from sklearn.model_selection import train_test_split

# Importamos modelo Random Forest
from sklearn.ensemble import RandomForestRegressor
# %%

raw = pd.read_csv('HORA.csv')
# %%
raw['date'] = pd.to_datetime(raw['TIM_DAY'])
raw['fechahora'] = pd.to_datetime(
    raw['date']) + raw['TIM_HOUR'].astype('timedelta64[h]')
# %%

# %%
raw.index = pd.date_range(start=raw.index.min(),
                          periods=len(raw), freq='60Min')
raw['COST_SHIFT_1H'] = raw['COST'].shift(periods=-1, freq='H')
raw['COST_SHIFT_2H'] = raw['COST'].shift(periods=-2, freq='H')
raw['COST_SHIFT_3H'] = raw['COST'].shift(periods=-3, freq='H')
raw['COST_SHIT_4H'] = raw['COST'].shift(periods=-4, freq='H')
raw.set_index('fechahora').dropna()
raw['CostoMañana'] = raw['COST'].shift(periods=1, freq='D')
raw.dropna(inplace=True)
#%%
raw['pctCOST'] = raw['COST'].pct_change()
raw['pctTROAS'] = raw['TROAS'].pct_change()
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.heatmap(raw.corr(), annot=True)
#%%
# %%
file = raw.copy().dropna()




## BORRAR

# analisis = raw.copy().dropna()
# analisis.set_index('fechahora',drop=True, inplace=True)

# costo =  analisis[['COST']]
# costo['mañana'] = costo['COST'].shift(periods=-1 , freq='D')
# costo = costo.dropna()

## BORRAR












#%%
# Calculamos un promedio del costo con un rolling window para tener de referencia
# ¿Cuánto mejor es nuestro modelo que tirar el promedio?
file['average'] = np.array(file.set_index(['TIM_DAY']).sort_index(ascending=False)[
                           'COST'].rolling(14).mean().sort_index(ascending=True).copy())

backup = file.copy()
file.drop(columns=['DIASEM', 'TIM_HOUR',
                   'ACCOUNT_NAME', 'TIM_DAY', 'date','fechahora'], inplace=True)
#%%
file.dropna(inplace=True)

# %%
# Usamos One-Hot encoding para las variables categoricas (en este caso dia semana)
file.drop(columns=['pctCOST'], inplace=True)

features = pd.get_dummies(file)

# %%

VarInteres = 'CostoMañana'

# labels : lo que queremos predecir, en este caso es el costo de Search
labels = np.array(features[VarInteres])
# Tenemos que sacar costo de los features
# usamos axis = 1 para dropear a nivel columna
features = features.drop(VarInteres, axis=1)
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







import plotly.express as px 

# Plot 
fig = px.line(raw, x='fechahora', y=raw['COST'].pct_change())

# Only thing I figured is - I could do this 
fig.add_scatter(x=raw['fechahora'], y=raw['TROAS'].pct_change()) # Not what is desired - need a line 

# Show plot 
fig.show()

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
