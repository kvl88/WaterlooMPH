from sklearn.datasets import make_regression
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sa
from yellowbrick.regressor import CooksDistance

fosfuel = pd.read_csv('MER_T01_03_fossilfuels.csv')
renew = pd.read_csv('MER_T01_03_fossilfuels.csv')
labprod = pd.read_csv('OPHNFB_laborprod.csv')
visualizer = CooksDistance()

#FOSSIL FUEL DATE CODE
fosfuel = fosfuel.drop(fosfuel.columns[[0,3,5]], axis = 1)
fosfuel.columns = ['Date', 'CperQuadBTU_ff', 'Type']
#YYYY13 is the annual total
#convert date column to string, filter for those ending in 13
#drop missing values
#convert back to int
fosfuel_date = fosfuel['Date'].apply(str)
fosfilter_13 = fosfuel_date.str.endswith('13')
fosfuel['Date'] = fosfuel_date[fosfilter_13]
fosfuel_13strip = fosfuel['Date']
fosfuel_13strip = fosfuel_13strip.str[:4]
#throw cleaned values into fos fuel date column
fosfuel['Date'] = fosfuel_13strip
#drop missing values in place
fosfuel.dropna(inplace=True)
#Filter by total fossil fuels consumption
fosfuel = fosfuel[fosfuel['Type'] == 'Total Fossil Fuels Consumption']
#drop the type column since we have filter in place
fosfuel = fosfuel.drop(fosfuel.columns[2], axis = 1)
#use date colum as index

#convert consumption data to int
fosfuel_typefix = pd.to_numeric(fosfuel['CperQuadBTU_ff'])
fosfuel['CperQuadBTU_ff'] = fosfuel_typefix
fosfuel = fosfuel.drop(fosfuel.columns[2:], axis = 1)
fosfuel = fosfuel.set_index(['Date'])

#RENEWABLES DATE CODE
renew = renew.drop(renew.columns[[0,3,5]], axis = 1)
renew.columns = ['Date', 'CperQuadBTU_renew', 'Type']
#YYYY13 is the annual total
#convert date column to string, filter for those ending in 13
#drop missing values
#convert back to int
renew_date = renew['Date'].apply(str)
renewfilter_13 = renew_date.str.endswith('13')
renew['Date'] = renew_date[renewfilter_13]
renew_13strip = renew['Date']
renew_13strip = renew_13strip.str[:4]
#throw cleaned values into fos fuel date column
renew['Date'] = renew_13strip
#drop missing values in place
renew.dropna(inplace=True)
#Filter by total fossil fuels consumption
renew = renew[renew['Type'] == 'Total Renewable Energy Consumption']
#drop the type column since we have filter in place
#renew = renew.drop(renew.columns[2], axis = 1)
#use date colum as index

#convert consumption data to int
renew_typefix = pd.to_numeric(renew['CperQuadBTU_renew'])
renew['CperQuadBTU_renew'] = renew_typefix
renew = renew.drop(renew.columns[2:], axis = 1)
renew = renew.set_index(['Date'])



labprod.columns = ['Date', 'OphAllWork']
labprod_datefix = labprod['Date']
labprod_datefix = labprod_datefix.str[:4]
labprod['Date'] = labprod_datefix
labprod = labprod.set_index(['Date'])

joined1 = fosfuel.join(labprod)
joined2 = renew.join(labprod)



model1 = smf.ols('OphAllWork ~ CperQuadBTU_ff', joined1).fit()
model2 = smf.ols('OphAllWork ~ CperQuadBTU_renew', joined2).fit()

print(model1.summary())
print(joined1.corr())

print(model2.summary())
print(joined2.corr())


#cooks distance diagnostics
#visualizer.fit(joined1['CperQuadBTU_ff'].values.reshape(-1, 1), joined1['OphAllWork'])
#joined1['Distance'] = visualizer.distance_
#joined1.sort_values('Distance', ascending = False).head()

#visualizer.fit(joined2['CperQuadBTU_renew'].values.reshape(-1, 1), joined2['OphAllWork'])
#joined2['Distance'] = visualizer.distance_
#joined2.sort_values('Distance', ascending = False).head()





sns.regplot(x=joined1['CperQuadBTU_ff'], y=joined1['OphAllWork'])
sns.regplot(x=joined2['CperQuadBTU_renew'], y=joined2['OphAllWork'])

#joined1.to_csv('FossilFuelvsLabourProd_clean.csv')
#joined2.to_csv('RenewvsLabourProd_clean.csv')
