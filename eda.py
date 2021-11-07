# %%
from pandas_profiling import ProfileReport
import pandas as pd
import matplotlib.pyplot as plt

# %%

raw = pd.read_csv('HORA.csv')
raw['date'] = pd.to_datetime(raw['TIM_DAY'])
raw['fechahora'] = pd.to_datetime(
    raw['date']) + raw['TIM_HOUR'].astype('timedelta64[h]')


# %%
analisis = raw.drop(
    columns=['ACCOUNT_NAME', 'TIM_DAY', 'DIASEM', 'TIM_HOUR', 'date']).copy()

prof = ProfileReport(analisis)
prof.to_file(output_file='output.html')
# Group By day and Sum Values
diario = analisis.groupby(pd.Grouper(key='fechahora', freq='D')).sum()
# %%
# EDA A NIVEL DIARIO
diario = raw.groupby('date').agg({'COST': 'sum', 'TROAS': 'mean'})
diario.plot(subplots=True)
# nos quedamos con las fechas en las que empiezan a haber cambios de target roas
diario = diario[(diario.index > '2021-08-02') & (diario.index < '2021-10-31')]
# %%
# GRAFICAMOS LOS HISTOGRAMAS
costos, troas = diario['COST'], diario['TROAS']
costos.plot.hist(bins=50, title='costo')
plt.show()
troas.plot.hist(bins=50, title='troas')
plt.show()
# %%
diario[['1d_costo_pct', '1d_troas_pct']
       ] = diario[['COST', 'TROAS']].pct_change()

# Shifteamos los valores

diario[['1d_cost_past', '1d_troas_past']] = diario[['COST', 'TROAS']].shift(-1)


# %%
plt.scatter(lng_df['5d_close_pct'], lng_df['5d_close_future_pct'])
plt.show()

####

# Definimos una columna como Target Roas Ponderado - Como TROAS x CLICKS  COMO PUJA TOTAL
# %%


def ArmarFechaHora(df1):

    df2 = df1
    df2['date'] = pd.to_datetime(df2['TIM_DAY'])
    df2['fechahora'] = pd.to_datetime(
        df2['date']) + df2['TIM_HOUR'].astype('timedelta64[h]')
    return df2

def puja(df1):

    df2 = df1
##    df2.set_index('')
    df2['PujaTotal'] = df2['TROAS']*df2['CLICKS']
    df2['ROASREAL'] = df2['CONVERSION_VALUE']/df2['COST']
    df2['SpreadDia'] = df2['TROAS'] * df2['CLICKS'] - df2['ROASREAL']
    return df2


def diairo(df1):
    ##df2 = df1.copy()

    df2 = pd.to_DataFrame(df1).groupby('TIM_DAY').agg({'TROAS':'mean','CLICKS':'sum','IMPRESSIONS':'sum','COST':'sum','CONVERSION_VALUE':'sum','PujaTotal':'sum','SpreadDia':'sum','ROASREAL':'mean'})
    return df2

def shiftValues(df1):
    df2 = df1.copy()
    
##    df2['']

# ROAS REAL = CONVERSION_VALUE / COST
#%%
hora = ArmarFechaHora(raw)
#%%
pujado =  puja(hora)
#%%
# dia  = diario(pujado)
dia = pujado.groupby('TIM_DAY').agg({'TROAS':'mean','CLICKS':'sum','IMPRESSIONS':'sum','COST':'sum','CONVERSION_VALUE':'sum','PujaTotal':'sum','SpreadDia':'sum','ROASREAL':'mean'})

#%%
completo = shiftValues(dia)

#%%
completo = dia.copy()
completo[['roasAyer']] =  completo['TORAS'].shift(1)

# POR DIA  =  (TARGET ROAS SETEADO x CLICKS) - (ROAS REAL X CLICKS)

# Target
# %%
