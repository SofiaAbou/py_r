import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_1samp, linregress
import statsmodels.api as sm

# Creazione dei dati
data = {
    'TERRITORIO': ['Piemonte', "Valle d'Aosta", 'Liguria', 'Lombardia', 'Trentino Alto Adige', 'Veneto', 
                   'Friuli Venezia Giulia', 'Emilia Romagna', 'Toscana', 'Umbria', 'Marche', 'Lazio', 
                   'Abruzzo', 'Molise', 'Campania', 'Puglia', 'Basilicata', 'Calabria', 'Sicilia', 'Sardegna'],
    'TEMPO DETERM.': [173, 7, 72, 394, 76, 236, 59, 255, 205, 49, 86, 317, 77, 15, 256, 205, 32, 102, 272, 82],
    'TEMPO INDETER.': [1234, 36, 407, 3241, 331, 1529, 361, 1345, 1051, 236, 415, 1590, 313, 61, 1026, 773, 117, 304, 845, 355],
    'TOTALE': [1407, 43, 479, 3635, 407, 1765, 420, 1600, 1256, 285, 501, 1907, 390, 76, 1282, 978, 149, 406, 1117, 437]
}

dati = pd.DataFrame(data)

# Visualizza le prime righe e i nomi delle colonne per verifica
print(dati.head())
print(dati.columns)

# Istogrammi
plt.figure(figsize=(8, 6))
plt.hist(dati['TEMPO DETERM.'], bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Tempo Determinato')
plt.ylabel('Frequenza')
plt.title('Distribuzione Contratti a Tempo Determinato')
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(dati['TEMPO INDETER.'], bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Tempo Indeterminato')
plt.ylabel('Frequenza')
plt.title('Distribuzione Contratti a Tempo Indeterminato')
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(dati['TOTALE'], bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Totale')
plt.ylabel('Frequenza')
plt.title('Distribuzione Totale')
plt.show()

# Diagrammi a Barre
plt.figure(figsize=(10, 6))
plt.bar(dati['TERRITORIO'], dati['TEMPO DETERM.'], color='green')
plt.xlabel('Regione')
plt.ylabel('Contratti Tempo Determinato')
plt.title('Contratti a Tempo Determinato per Regione')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Grafici a Torta
plt.figure(figsize=(8, 8))
plt.pie(dati['TEMPO DETERM.'], labels=dati['TERRITORIO'], autopct='%1.1f%%', startangle=140)
plt.title('Percentuale Contratti a Tempo Determinato per Regione')
plt.show()

# Boxplot
plt.figure(figsize=(8, 6))
plt.boxplot([dati['TEMPO DETERM.'], dati['TEMPO INDETER.'], dati['TOTALE']], labels=['Tempo Determinato', 'Tempo Indeterminato', 'Totale'])
plt.title('Distribuzione dei Contratti')
plt.ylabel('Numero di Contratti')
plt.show()

# Test Statistici
f_stat, p_value = f_oneway(dati['TEMPO DETERM.'], dati['TEMPO INDETER.'])
print(f"F statistic: {f_stat}, p-value: {p_value}")

t_stat, p_value = ttest_1samp(dati['TEMPO DETERM.'], popmean=dati['TEMPO DETERM.'].mean())
print(f"T statistic: {t_stat}, p-value: {p_value}")

# Correlazione
correlation_matrix = dati[['TEMPO DETERM.', 'TEMPO INDETER.', 'TOTALE']].corr()
print(correlation_matrix)

# Regressione Lineare Semplice
slope, intercept, r_value, p_value, std_err = linregress(dati['TEMPO DETERM.'], dati['TOTALE'])
print(f"Slope: {slope}, Intercept: {intercept}, R-squared: {r_value**2}, p-value: {p_value}")

plt.figure(figsize=(8, 6))
plt.scatter(dati['TEMPO DETERM.'], dati['TOTALE'])
plt.plot(dati['TEMPO DETERM.'], intercept + slope * dati['TEMPO DETERM.'], 'r', label='Regression Line')
plt.xlabel('Tempo Determinato')
plt.ylabel('Totale Contratti')
plt.title('Regressione Lineare')
plt.legend()
plt.show()

# Regressione Lineare Multipla
X = dati[['TEMPO DETERM.', 'TEMPO INDETER.']]
y = dati['TOTALE']
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
print(model.summary())
