from sklearn.datasets import make_regression
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sa
from yellowbrick.regressor import CooksDistance
from functools import reduce


Cancerdata = pd.read_csv('cdc_lungcancerdata_indianawashington.csv')
Copddata = pd.read_csv('cdc_copddata_indianawashington.csv')
WAemission = pd.read_csv('washington_GHGemissions_data.csv')
WAenergy = pd.read_csv('washington_energyconsumpdata.csv', skiprows = 75)
IAemission = pd.read_csv('indiana_GHGemissions_data.csv')
IAenergy = pd.read_csv('indiana_energyconsumption.csv', skiprows = 75)


#cleaning the combined washington and indiana cancer data 
Cancerdata = Cancerdata.drop(Cancerdata.columns[[0,2,4,5,6,8]], axis = 1)
Cancerdata.dropna(inplace=True)
CCdatefix = Cancerdata['Year'].apply(str)
CCdatefix = CCdatefix.str[:4]
Cancerdata['Year'] = CCdatefix

#wash cancer data
CancerdataWA_filter = Cancerdata[Cancerdata['State'] == "Washington"]
CancerdataWA = CancerdataWA_filter
CancerdataWA = CancerdataWA.drop(columns='State')
CancerdataWA['Age-Adjusted Rate'] = CancerdataWA['Age-Adjusted Rate'].astype(int)
CancerdataWA.rename(columns = {'Age-Adjusted Rate':'CanDeathPer100k'}, inplace=True)

#Indiana Cancer data
CancerdataIA_filter = Cancerdata[Cancerdata['State'] == "Indiana"]
CancerdataIA = CancerdataIA_filter
CancerdataIA = CancerdataIA.drop(columns='State')
CancerdataIA['Age-Adjusted Rate'] = CancerdataIA['Age-Adjusted Rate'].astype(int)
CancerdataIA.rename(columns = {'Age-Adjusted Rate':'CanDeathPer100k'}, inplace=True)

#cleaning the combined washington and indiana COPD data
Copddata_total = Copddata[Copddata['Notes'] == 'Total']
Copddata_total = Copddata_total.drop(Copddata.columns[[0,2,4,5,6,7,8,9,11]], axis = 1)
Copddata_total.dropna(inplace=True)
CCdatefix = Copddata_total['Year'].apply(str)
CCdatefix = CCdatefix.str[:4]
Copddata_total['Year'] = CCdatefix



CopddataWA_total_filter = Copddata_total[Copddata_total['State'] == "Washington"]
CopddataWA_total = CopddataWA_total_filter
CopddataWA_total = CopddataWA_total.drop(columns='State')
CopddataWA_total['Age Adjusted Rate'] = CopddataWA_total['Age Adjusted Rate'].astype(int)
CopddataWA_total.rename(columns = {'Age Adjusted Rate':'COPDeathPer100k'}, inplace=True)

CopddataIA_total_filter = Copddata_total[Copddata_total['State'] == "Indiana"]
CopddataIA_total = CopddataIA_total_filter
CopddataIA_total = CopddataIA_total.drop(columns='State')
CopddataIA_total['Age Adjusted Rate'] = CopddataIA_total['Age Adjusted Rate'].astype(int)
CopddataIA_total.rename(columns = {'Age Adjusted Rate':'COPDeathPer100k'}, inplace=True)

#cleaning Washington Emission Data
WAemission.drop([0,1,2,3,4,6], axis=0, inplace=True)
WAemission.rename(columns = {'Washington Emissions by Gas, MMT CO2 eq.':'TotalNetEmissions'}, inplace=True)
WAemissionMELT = WAemission.melt(id_vars = ['TotalNetEmissions'])
WAemission_total = WAemissionMELT
WAemission_total = WAemission_total.drop(WAemission_total.columns[[0]], axis = 1)
WAemission_total.columns = ['Year', 'TotalNetEmissions']
WAemission_total['TotalNetEmissions'] = WAemission_total['TotalNetEmissions'].astype(int)

#cleaning Indiana Emission Data
IAemission.drop([0,1,2,3,4,6], axis=0, inplace=True)
IAemission.rename(columns = {'Indiana Emissions by Gas, MMT CO2 eq.':'TotalNetEmissions'}, inplace=True)
IAemissionMELT = IAemission.melt(id_vars = ['TotalNetEmissions'])
IAemission_total = IAemissionMELT
IAemission_total = IAemission_total.drop(IAemission_total.columns[[0]], axis = 1)
IAemission_total.columns = ['Year', 'TotalNetEmissions']
IAemission_total['TotalNetEmissions'] = IAemission_total['TotalNetEmissions'].astype(int)

#cleaning Washington consumption data
WAenergy = WAenergy.drop(list(WAenergy)[13:], axis=1)
WAenergy = WAenergy.drop(list(WAenergy)[1:12], axis=1)
WAenergy.columns = ['Year', 'ConsPerTrilBTU']
WAenergy.dropna(inplace = True)
WAenergy['ConsPerTrilBTU'] = WAenergy['ConsPerTrilBTU'].astype(int)

#cleaning Indiana consumption data
IAenergy = IAenergy.drop(list(IAenergy)[14:], axis=1)
IAenergy = IAenergy.drop(list(IAenergy)[1:13], axis=1)
IAenergy.columns = ['Year', 'ConsPerTrilBTU']
IAenergy.dropna(inplace = True)
IAenergy['ConsPerTrilBTU'] = IAenergy['ConsPerTrilBTU'].astype(int)
IAenergy['Year'] = IAenergy['Year'].astype(str)


#this allows me to merge multiple dataframes together on their index ('Year)
#using a list of dataframes
IAdata = [IAemission_total, IAenergy, CancerdataIA, CopddataIA_total]
IAjoined = reduce(lambda x,y: x.merge(y,on='Year', how='left'), IAdata)
IAjoined = IAjoined.set_index('Year')
IAjoined.dropna(inplace=True)

WAdata = [WAemission_total, WAenergy, CancerdataWA, CopddataWA_total]
WAjoined = reduce(lambda x,y: x.merge(y,on='Year', how='left'), WAdata)
WAjoined = WAjoined.set_index('Year')
WAjoined.dropna(inplace=True)



CanModel_WA = smf.ols("CanDeathPer100k ~ TotalNetEmissions + ConsPerTrilBTU", WAjoined).fit()
print(CanModel_WA.summary())
COPDModel_WA = smf.ols("COPDeathPer100k ~ TotalNetEmissions + ConsPerTrilBTU", WAjoined).fit()
print(COPDModel_WA.summary())
print(WAjoined.corr())

CanModel_IA = smf.ols("CanDeathPer100k ~ TotalNetEmissions + ConsPerTrilBTU", IAjoined).fit()
print(CanModel_IA.summary())
COPDModel_IA = smf.ols("COPDeathPer100k ~ TotalNetEmissions + ConsPerTrilBTU", IAjoined).fit()
print(COPDModel_IA.summary())
print(IAjoined.corr())

plt.figure()
plt.title('Emissions Vs Cancer Death Rate Per 100K - Washington')
sns.regplot(x=WAjoined['CanDeathPer100k'], y=WAjoined['TotalNetEmissions'])
plt.figure()
plt.title('Emissions Vs COPD Death Rate Per 100K - Washington')
sns.regplot(x=WAjoined['COPDeathPer100k'], y=WAjoined['TotalNetEmissions'])
plt.figure()
plt.title('Emissions Vs Cancer Death Rate Per 100K - Indiana')
sns.regplot(x=IAjoined['CanDeathPer100k'], y=IAjoined['TotalNetEmissions'])
plt.figure()
plt.title('Emissions Vs COPD Death Rate Per 100K - Indiana')
sns.regplot(x=IAjoined['COPDeathPer100k'], y=IAjoined['TotalNetEmissions'])

#IAjoined.to_csv('IndianaData_clean.csv')
#WAjoined.to_csv('WashingtonData_clean.csv')

