# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 13:45:43 2023

@author: tomgd
"""

import pandas as pd

# importing in data for energy
df  = pd.read_csv('MER_T02_01A.csv')
#converting value column into numeric
df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

# create new columns based on sorted values
for val in df['MSN'].unique():
    col_name = f'{val}'
    df[col_name] = df.loc[df['MSN'] == val, 'Value']
    df = df.fillna(0)
    
# extracting relevant columns and removing empty rows

residential = df['TERCBUS']
residential = residential.drop(residential[residential == 0].index)
residential = residential.reset_index(drop=True)

commercial = df['TECCBUS']
commercial = commercial.drop(commercial[commercial == 0].index)
commercial = commercial.reset_index(drop=True)

industrial = df['TEICBUS']
industrial = industrial.drop(industrial[industrial == 0].index)
industrial = industrial.reset_index(drop=True)

# combining extracted columns and cleaning data (removing first 24 rows and every 13th row)

dat = pd.DataFrame({'residential': residential, 'commercial': commercial, 'industrial': industrial})
dat = dat.drop(index=range(24))
dat = dat.drop(dat.index[12::13])
dat = dat.reset_index(drop=True)

# converting data to time indexed and setting it to quarterly instead of monthly to align with labour productivity data
date_range = pd.date_range(start='1973-01-01', end='2022-11-01', freq='M')
dat['Date'] = date_range
dat.set_index('Date', inplace=True)
quarterly_data = dat.resample('Q').sum()

# importing in Labour productivity data and adding it to the dataframe with the energy data
labour = pd.read_csv('PRS85006091.csv', index_col='DATE')
labour.index = pd.to_datetime(labour.index)
labour = labour['PRS85006091']
labour = labour.reindex(quarterly_data.index, method='nearest')
quarterly_data['labour'] = labour

# Preliminary data analysis and regression model.

import statsmodels.formula.api as smf

model1 = smf.ols(formula = 'residential ~ labour', data =
quarterly_data).fit()
print(model1.summary())

model2 = smf.ols(formula = 'commercial ~ labour', data =
quarterly_data).fit()
print(model2.summary())

model3 = smf.ols(formula = 'industrial ~ labour', data =
quarterly_data).fit()
print(model3.summary())

import matplotlib.pyplot as plt


# plot data
quarterly_data.plot(y='residential')
quarterly_data.plot(y='commercial')
quarterly_data.plot(y='industrial')
quarterly_data.plot(y='labour')

plt.show()

# Correlation

print(quarterly_data.corr())

# summary statistics

print(quarterly_data['residential'].describe())
print(quarterly_data['commercial'].describe())
print(quarterly_data['industrial'].describe())
print(quarterly_data['labour'].describe())


# incorporating health data
health  = pd.read_csv('health.csv')
date_range2 = pd.date_range(start='1980-01-01', end='2019-01-01', freq='A')
health['Date'] = date_range2
health = health.drop(index=38)

#converting previous data to match with 
annual_data = quarterly_data.resample('A').sum()
annual_data = annual_data.drop(annual_data.loc['1973':'1979'].index)
annual_data = annual_data.drop(annual_data.loc['2018':'2022'].index)
annual_data = annual_data.reset_index(drop=True)
merged_annual = pd.merge(annual_data, health, left_index=True, right_index=True)

merged_annual['Date'] = pd.to_datetime(merged_annual['Date'])
annual = merged_annual.set_index('Date')

# regression with health data

model3 = smf.ols(formula = 'industrial ~ labour', data =
quarterly_data).fit()



# plot data
annual.plot(y='residential')
annual.plot(y='commercial')
annual.plot(y='industrial')
annual.plot(y='labour')
annual = annual.rename(columns={'Chronic lower respiratory diseases6/,7/':'health'})

plt.show()

print(annual.dtypes)
model4 = smf.ols(formula = 'health ~ labour + residential', data =
annual).fit()
print(model4.summary())

model5 = smf.ols(formula = 'health ~ labour + commercial', data =
annual).fit()
print(model5.summary())

model6 = smf.ols(formula = 'health ~ labour + industrial', data =
annual).fit()
print(model6.summary())


# regression with interaction term

model7 = smf.ols(formula = 'health ~ labour + residential + labour*residential', data =
annual).fit()
print(model7.summary())

model8 = smf.ols(formula = 'health ~ labour + commercial + labour*commercial', data =
annual).fit()
print(model8.summary())

model9 = smf.ols(formula = 'health ~ labour + industrial + labour*industrial', data =
annual).fit()
print(model9.summary())
