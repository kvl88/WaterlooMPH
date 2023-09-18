
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sa
from functools import reduce


fosfuel = pd.read_csv('MER_T01_03_fossilfuels.csv')
renew = pd.read_csv('MER_T01_03_fossilfuels.csv')
labprod = pd.read_csv('OPHNFB_laborprod.csv')
Cancerdata = pd.read_csv('cdc_lungcancerdata_indianawashington.csv')
Copddata = pd.read_csv('cdc_copddata_indianawashington.csv')
WAemission = pd.read_csv('washington_GHGemissions_data.csv')
WAenergy = pd.read_csv('washington_energyconsumpdata.csv', skiprows = 75)
IAemission = pd.read_csv('indiana_GHGemissions_data.csv')
IAenergy = pd.read_csv('indiana_energyconsumption.csv', skiprows = 75)
df  = pd.read_csv('MER_T02_01A.csv')
health  = pd.read_csv('health.csv')



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

# incorporating health data

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
merged_annual = merged_annual.rename(columns={'Chronic lower respiratory diseases6/,7/':'health', 'Date':'Year'})
merged_annual_datefix = merged_annual['Year']
merged_annual_datefix = merged_annual_datefix.astype(str)
merged_annual_datefix = merged_annual_datefix.str[:4]
merged_annual['Year'] = merged_annual_datefix
annual = merged_annual.set_index('Year')

# regression with health data

# plot data
annual.plot(y='residential')
annual.plot(y='commercial')
annual.plot(y='industrial')
annual.plot(y='labour')


plt.show()

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

#FOSSIL FUEL DATE CODE
fosfuel = fosfuel.drop(fosfuel.columns[[0,3,5]], axis = 1)
fosfuel.columns = ['Year', 'CperQuadBTU_ff', 'Type']
#YYYY13 is the annual total
#convert date column to string, filter for those ending in 13
#drop missing values
#convert back to int
fosfuel_date = fosfuel['Year'].apply(str)
fosfilter_13 = fosfuel_date.str.endswith('13')
fosfuel['Year'] = fosfuel_date[fosfilter_13]
fosfuel_13strip = fosfuel['Year']
fosfuel_13strip = fosfuel_13strip.str[:4]
#throw cleaned values into fos fuel date column
fosfuel['Year'] = fosfuel_13strip
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
fosfuel = fosfuel.set_index(['Year'])

#RENEWABLES DATE CODE
renew = renew.drop(renew.columns[[0,3,5]], axis = 1)
renew.columns = ['Year', 'CperQuadBTU_renew', 'Type']
#YYYY13 is the annual total
#convert date column to string, filter for those ending in 13
#drop missing values
#convert back to int
renew_date = renew['Year'].apply(str)
renewfilter_13 = renew_date.str.endswith('13')
renew['Year'] = renew_date[renewfilter_13]
renew_13strip = renew['Year']
renew_13strip = renew_13strip.str[:4]
#throw cleaned values into fos fuel date column
renew['Year'] = renew_13strip
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
renew = renew.set_index(['Year'])



labprod.columns = ['Year', 'OphAllWork']
labprod_datefix = labprod['Year']
labprod_datefix = labprod_datefix.str[:4]
labprod['Year'] = labprod_datefix
labprod = labprod.set_index('Year')

joined1 = fosfuel.join(labprod)
joined2 = renew.join(labprod)



model1 = smf.ols('OphAllWork ~ CperQuadBTU_ff' , joined1).fit()
model2 = smf.ols('OphAllWork ~ CperQuadBTU_renew', joined2).fit()

print(model1.summary())
print(joined1.corr())

print(model2.summary())
print(joined2.corr())


ax = sns.regplot(x="CperQuadBTU_ff", y ="OphAllWork",
                 data=joined1, fit_reg=True,
                 scatter_kws={'alpha': 0.5, 's': 20},
                 line_kws={'alpha': 0.8, 'linewidth': 2, 'label': 'Fossil Fuels'},
                 color='Black',  
                 x_jitter=.2, order=2)

ax = sns.regplot(x="CperQuadBTU_renew", y ="OphAllWork",
                 data=joined2, fit_reg=True,
                 scatter_kws={'alpha': 0.8, 's': 20},
                 line_kws={'alpha': 0.5, 'linewidth': 2, 'label': 'Renewables'},
                 color='Green',
                 x_jitter=.2, order=2)
plt.xlabel('Consumption per Quadrillion BTU')
plt.ylabel('Labor Productivity - Output per Hour')
ax.legend(loc='upper center', borderpad=.2)
plt.grid(ax)
plt.show()


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
plt.grid()
plt.title('Emissions Vs Cancer Death Rate Per 100K - Washington')
sns.regplot(x=WAjoined['CanDeathPer100k'], y=WAjoined['TotalNetEmissions'])
plt.figure()
plt.grid()
plt.title('Emissions Vs COPD Death Rate Per 100K - Washington')
sns.regplot(x=WAjoined['COPDeathPer100k'], y=WAjoined['TotalNetEmissions'])
plt.figure()
plt.grid()
plt.title('Emissions Vs Cancer Death Rate Per 100K - Indiana')
sns.regplot(x=IAjoined['CanDeathPer100k'], y=IAjoined['TotalNetEmissions'])
plt.figure()
plt.grid()
plt.title('Emissions Vs COPD Death Rate Per 100K - Indiana')
sns.regplot(x=IAjoined['COPDeathPer100k'], y=IAjoined['TotalNetEmissions'])


cleanmerged = [annual, joined1, joined2, WAjoined, IAjoined]
cleanmerged = reduce(lambda x,y: x.merge(y,on='Year', how='left'), cleanmerged)
cleanmerged.to_csv('627_datamerge_clean.csv')