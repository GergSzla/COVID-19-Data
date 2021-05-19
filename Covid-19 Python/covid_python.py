from pandas import Series, DataFrame
import pandas 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.cbook as cbook
import seaborn
from scipy.interpolate import make_interp_spline, BSpline
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans

#import files
cov_world_dat, cov_irl_county_cases, cov_pcd_shipment  = None, None, None
with pandas.ExcelFile("covid-data.xlsx") as reader:
    cov_world_dat = pandas.read_excel(reader, sheet_name='cov_world_dat') #get sheet 1
    cov_irl_county_cases = pandas.read_excel(reader, sheet_name='cov_irl_county_cases') #get sheet 2
    cov_pcd_shipment = pandas.read_excel(reader, sheet_name='pcd_shipments') #get sheet 3


#convert all columns to uppercase
cov_world_dat.columns = map(str.upper, cov_world_dat.columns)
cov_irl_county_cases.columns = map(str.upper, cov_irl_county_cases.columns)
cov_pcd_shipment.columns = map(str.upper, cov_pcd_shipment.columns)

#change to numeric
cov_world_dat['TOTAL_CASES'] = pandas.to_numeric(cov_world_dat['TOTAL_CASES'])
cov_world_dat['NEW_CASES'] = pandas.to_numeric(cov_world_dat['NEW_CASES'])
cov_world_dat['NEW_TESTS'] = pandas.to_numeric(cov_world_dat['NEW_TESTS'])
cov_world_dat['NEW_DEATHS'] = pandas.to_numeric(cov_world_dat['NEW_DEATHS'])
cov_world_dat['TOTAL_DEATHS'] = pandas.to_numeric(cov_world_dat['TOTAL_DEATHS'])


##QUESTION ONE

is_irl = cov_world_dat.ISO_CODE == "IRL"
sub2 = cov_world_dat[is_irl]
irl_sub = sub2[['ISO_CODE', 'LOCATION','DATE','TOTAL_CASES','NEW_CASES','NEW_VACCINATIONS_SMOOTHED','NEW_DEATHS','NEW_TESTS','TOTAL_DEATHS', 'TOTAL_CASES_PER_MILLION']]

#Calculate new column INFECTED DEATH RATIO TOTAL
irl_sub['INFECTED_DEATH_RATIO_TOTAL'] = irl_sub['TOTAL_CASES'] / irl_sub['TOTAL_DEATHS']

irl_sub['INFECTED_DEATH_RATIO_TOTAL'] = pandas.to_numeric(irl_sub['INFECTED_DEATH_RATIO_TOTAL'])

pcd_sub = cov_pcd_shipment[['DATE','WORKSTATION','SLATE_TABLET','NOTEBOOK','DETACHABLE_TABLET','DESKTOP']].fillna(0)

#get data county data from sheet 2
leinster_sub = cov_irl_county_cases[['DATE','CARLOW','WEXFORD','KILKENNY','WICKLOW','LONGFORD','WESTMEATH','MEATH','LOUTH','DUBLIN','OFFALY','LAOIS','KILDARE']]
ulster_sub = cov_irl_county_cases[['DATE','DONEGAL','CAVAN','MONAGHAN']]
connacht_sub = cov_irl_county_cases[['DATE','LEITRIM','SLIGO','MAYO','ROSCOMMON','GALWAY']]
munster_sub = cov_irl_county_cases[['DATE','WATERFORD','TIPPERARY','CLARE','LIMERICK','CORK','KERRY']]

#Describe Data for irl

print("Data for: IRELAND")
print(irl_sub)

print('Describe Number of New Cases')
desc1 = irl_sub['NEW_CASES'].describe()
print(desc1)

print('mean')
mean1 = irl_sub['NEW_CASES'].mean()
print(mean1)

print('std deviation')
std1 = irl_sub['NEW_CASES'].std()
print(std1)

print('min')
min1 = irl_sub['NEW_CASES'].min()
print(min1)

print('max')
max1 = irl_sub['NEW_CASES'].max()
print(max1)

print('median')
median1 = irl_sub['NEW_CASES'].median()
print(median1)

print('mode')
mode1 = irl_sub['NEW_CASES'].mode()
print(mode1)


#Sort New Cases Irl Freq Dist
print('Frequency Dist: New Cases (IRL)')
irl_sub['NEW_CASES_BINS'] = pandas.cut(irl_sub.NEW_CASES, [0,35,70,105,140,175,210,245,280,315,350,385,420,455,490,525], labels=['0-34','35-69','70-104','105-139','140-174','175-209','210-244','245-279','280-314','315-349','350-384','385-419','420-454','455-489','490-525'])
c4 = irl_sub['NEW_CASES_BINS'].value_counts(sort=False,dropna=True)
print(c4)


print('Describe Number of Infected Death Ratio Total')
desc2 = irl_sub['INFECTED_DEATH_RATIO_TOTAL'].describe()
print(desc2)

#Sort Infected Death Ratio Total Irl Freq Dist
print('Frequency Dist: Infected Death Ratio Total (IRL)')
irl_sub['INFECTED_DEATH_RATIO_TOTAL_BINS'] = pandas.cut(irl_sub.INFECTED_DEATH_RATIO_TOTAL, [0,27,54,81,108,135,162,189,216,243,270], labels=['<=27','<=54','<=81','<=108','<=135','<=162','<=189','<=216','<=243','<=270'])
c5 = irl_sub['INFECTED_DEATH_RATIO_TOTAL_BINS'].value_counts(sort=False,dropna=True)
print(c5)


print('Describe Number of New Deaths (IRL)')
desc2 = irl_sub['NEW_DEATHS'].describe()
print(desc2)

#Sort New Deaths Irl Freq Dist
print('Frequency Dist: New Deaths (IRL)')
irl_sub['NEW_DEATHS_BINS'] = pandas.cut(irl_sub.NEW_DEATHS, [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220], 
                                        labels=['<=10','<=20','<=30','<=40','<=50','<=60','<=70','<=80','<=90','<=100'
                                                ,'<=110','<=120','<=130','<=140','<=150','<=160','<=170','<=180','<=190','<=200',
                                                '<=210','<=220'])
c4 = irl_sub['NEW_DEATHS_BINS'].value_counts(sort=False,dropna=True)
print(c4)

#bar charts for freq dist.
bc1 = seaborn.countplot(x='NEW_DEATHS_BINS',data=irl_sub)
plt.xlabel('New Deaths For 2020 ')
plt.title('New Deaths Grouped For 2020 (IRL)')
plt.show()

bc1 = seaborn.countplot(x='INFECTED_DEATH_RATIO_TOTAL_BINS',data=irl_sub)
plt.xlabel('Infected Death Ratio Total For 2020 ')
plt.title('Infected Death Ratio Total Grouped For 2020 (IRL)')
plt.show()

bc1 = seaborn.countplot(x='NEW_CASES_BINS',data=irl_sub)
plt.xlabel('New Cases For 2020 ')
plt.title('New Cases Grouped For 2020 (IRL)')
plt.show()

#get data for NZL
is_nzl = cov_world_dat.ISO_CODE == "NZL"
sub2 = cov_world_dat[is_nzl]
nzl_sub = sub2[['ISO_CODE', 'LOCATION','DATE','TOTAL_CASES','NEW_CASES','NEW_VACCINATIONS_SMOOTHED','NEW_DEATHS','NEW_TESTS','TOTAL_DEATHS','TOTAL_CASES_PER_MILLION']].fillna(0)


nzl_sub['INFECTED_DEATH_RATIO_TOTAL'] = nzl_sub['TOTAL_CASES'].fillna(0) / nzl_sub['TOTAL_DEATHS'].fillna(0)
nzl_sub['INFECTED_DEATH_RATIO_TOTAL'] = pandas.to_numeric(nzl_sub['INFECTED_DEATH_RATIO_TOTAL'])


print("Data for: NEW ZEALAND")
print(nzl_sub)


## LINE GRAPH FOR IRL <AND> NZL 
x_labels = irl_sub['DATE']
y_cases_tot_irl = irl_sub['TOTAL_CASES']
y_cases_tot_nzl = nzl_sub['TOTAL_CASES']


plt.plot(x_labels, y_cases_tot_irl,color='green')
plt.plot(x_labels, y_cases_tot_nzl,color='orange')

plt.title('TOTAL COVID-19 CASES')
plt.xlabel('DATE')
plt.ylabel('CASES')


green_patch = mpatches.Patch(color='green', label='IRELAND')
orange_patch = mpatches.Patch(color='orange', label='NEW ZEALAND')

plt.legend(handles=[green_patch,orange_patch])
plt.grid(True)
plt.show()



green_patch = mpatches.Patch(color='green', label='IRELAND')
orange_patch = mpatches.Patch(color='orange', label='NEW ZEALAND')

plt.legend(handles=[green_patch,orange_patch])
plt.grid(True)
plt.show()

## LINE GRAPH FOR IRL CASES INC.
x_labels = irl_sub['DATE']
y_cases_tot_irl = irl_sub['NEW_CASES']
plt.plot(x_labels, y_cases_tot_irl,color='green')

plt.title('NEW COVID-19 CASES (IRL)')
plt.xlabel('DATE')
plt.ylabel('NEW CASES')

green_patch = mpatches.Patch(color='green', label='IRELAND')
plt.legend(handles=[green_patch])
plt.grid(True)
plt.show()





## LINE GRAPH FOR IRL CASES INC.
x_labels = nzl_sub['DATE']
y_cases_tot_nzl = nzl_sub['NEW_CASES']
plt.plot(x_labels, y_cases_tot_nzl,color='orange')

plt.title('NEW COVID-19 CASES (NZL)')
plt.xlabel('DATE')
plt.ylabel('NEW CASES')

orange_patch = mpatches.Patch(color='orange', label='NEW ZEALAND')
plt.legend(handles=[orange_patch])
plt.grid(True)
plt.show()



## LINE GRAPH FOR VACCINATED:CASES RATE IRL
x_labels = irl_sub['DATE']
y_cases_new_irl = irl_sub['NEW_CASES']
y_vaccinations_new_irl = irl_sub['NEW_VACCINATIONS_SMOOTHED']

plt.plot(x_labels, y_cases_new_irl,color='red')
plt.plot(x_labels, y_vaccinations_new_irl,color='blue')

plt.title('NEW COVID-19 CASES:VACCINATIONS (IRL)')
plt.xlabel('DATE')
plt.ylabel('NEW CASES & VACCINATIONS')

red_patch = mpatches.Patch(color='red', label='NEW CASES')
blue_patch = mpatches.Patch(color='blue', label='NEW VACCINATIONS [SMOOTHED]')

plt.legend(handles=[red_patch,blue_patch])
plt.grid(True)
plt.show()

## LINE GRAPH FOR VACCINATED:CASES RATE NZL
x_labels = nzl_sub['DATE']
y_cases_new_nzl = nzl_sub['NEW_CASES']
y_vaccinations_new_nzl = nzl_sub['NEW_VACCINATIONS_SMOOTHED']

plt.plot(x_labels, y_cases_new_nzl,color='red')
plt.plot(x_labels, y_vaccinations_new_nzl,color='blue')

plt.title('NEW COVID-19 CASES:VACCINATIONS (NZL)')
plt.xlabel('DATE')
plt.ylabel('NEW CASES & VACCINATIONS')

red_patch = mpatches.Patch(color='red', label='NEW CASES')
blue_patch = mpatches.Patch(color='blue', label='NEW VACCINATIONS [SMOOTHED]')

plt.legend(handles=[red_patch,blue_patch])
plt.grid(True)
plt.show()

#COMPARE TO GERMANY
#Infection - Death Rate Comp
is_deu = cov_world_dat.ISO_CODE == "DEU"
sub2 = cov_world_dat[is_deu]
deu_sub = sub2[['ISO_CODE', 'LOCATION','DATE','TOTAL_CASES','NEW_CASES','NEW_VACCINATIONS_SMOOTHED','NEW_DEATHS','NEW_TESTS','TOTAL_DEATHS','TOTAL_CASES_PER_MILLION']].fillna(0)

deu_sub['INFECTED_DEATH_RATIO_TOTAL'] = deu_sub['TOTAL_CASES'].fillna(0) / deu_sub['TOTAL_DEATHS'].fillna(0)
deu_sub['INFECTED_DEATH_RATIO_TOTAL'] = pandas.to_numeric(deu_sub['INFECTED_DEATH_RATIO_TOTAL'])


print("Data for: GERMANY")
print(deu_sub)

## LINE GRAPH FOR IRL <AND> DEU 
x_labels = irl_sub['DATE']
y_cases_tot_irl = irl_sub['TOTAL_CASES']
y_cases_tot_deu = deu_sub['TOTAL_CASES']


plt.plot(x_labels, y_cases_tot_irl,color='green')
plt.plot(x_labels, y_cases_tot_deu,color='black')

plt.title('TOTAL COVID-19 CASES')
plt.xlabel('DATE')
plt.ylabel('CASES')


green_patch = mpatches.Patch(color='green', label='IRELAND')
black_patch = mpatches.Patch(color='black', label='GERMANY')

plt.legend(handles=[green_patch,black_patch])
plt.grid(True)
plt.show()

## LINE GRAPH FOR IRL <AND> DEU 
x_labels = irl_sub['DATE']
y_cases_tot_irl = irl_sub['TOTAL_CASES_PER_MILLION']
y_cases_tot_deu = deu_sub['TOTAL_CASES_PER_MILLION']


plt.plot(x_labels, y_cases_tot_irl,color='green')
plt.plot(x_labels, y_cases_tot_deu,color='black')

plt.title('TOTAL COVID-19 CASES (/Million)')
plt.xlabel('DATE')
plt.ylabel('CASES')


green_patch = mpatches.Patch(color='green', label='IRELAND')
black_patch = mpatches.Patch(color='black', label='GERMANY')

plt.legend(handles=[green_patch,black_patch])
plt.grid(True)
plt.show()

## LINE GRAPH FOR IRL CASES INC.
x_labels = deu_sub['DATE']
y_cases_tot_deu = deu_sub['NEW_CASES']
plt.plot(x_labels, y_cases_tot_deu,color='black')

plt.title('NEW COVID-19 CASES (DEU)')
plt.xlabel('DATE')
plt.ylabel('NEW CASES')

black_patch = mpatches.Patch(color='black', label='GERMANY')
plt.legend(handles=[black_patch])
plt.grid(True)
plt.show()

## LINE GRAPH FOR VACCINATED:CASES RATE DEU
x_labels = deu_sub['DATE']
y_cases_new_deu = deu_sub['NEW_CASES']
y_vaccinations_new_deu = deu_sub['NEW_VACCINATIONS_SMOOTHED']

plt.plot(x_labels, y_cases_new_deu,color='red')
plt.plot(x_labels, y_vaccinations_new_deu,color='blue')

plt.title('NEW COVID-19 CASES:VACCINATIONS (DEU)')
plt.xlabel('DATE')
plt.ylabel('NEW CASES & VACCINATIONS')

red_patch = mpatches.Patch(color='red', label='NEW CASES')
blue_patch = mpatches.Patch(color='blue', label='NEW VACCINATIONS [SMOOTHED]')

plt.legend(handles=[red_patch,blue_patch])
plt.grid(True)
plt.show()


## SCATTER GRAPH FOR IRELAND (TESTS/DEATHS)
x = irl_sub['NEW_DEATHS']
y = irl_sub['NEW_TESTS']

plt.xlabel('New Deaths')
plt.ylabel('New Tests')
plt.title('Scatterplot for IRL DEATH:TEST')

plt.scatter(x, y, color = 'g')
plt.show()

## LINE GRAPH FOR IRL <AND> DEU for INFECTED_DEATH_RATIO_TOTAL 
x_labels = irl_sub['DATE']
y_cases_tot_irl = irl_sub['INFECTED_DEATH_RATIO_TOTAL']
y_cases_tot_deu = deu_sub['INFECTED_DEATH_RATIO_TOTAL']

plt.plot(x_labels, y_cases_tot_irl,color='green')
plt.plot(x_labels, y_cases_tot_deu,color='black')

plt.title('Total Cases:Total Deaths Ratio (IRL & DEU)')
plt.xlabel('DATE')
plt.ylabel('C:D Ratio')


green_patch = mpatches.Patch(color='green', label='IRELAND')
black_patch = mpatches.Patch(color='black', label='GERMANY')

plt.legend(handles=[green_patch,black_patch])
plt.grid(True)
plt.show()

## SCATTER GRAPH FOR IRELAND (TESTS/DEATHS)
x = irl_sub['NEW_CASES']
y = irl_sub['TOTAL_CASES']

plt.xlabel('Date')
plt.ylabel('New Cases')
plt.title('Regression Analysis for IRL New Cases')

plt.scatter(x, y, color = 'g')
plt.show()

#scatter plot for NEW_CASES : TOTAL_CASES
scat1 = seaborn.regplot(x="NEW_CASES", y="TOTAL_CASES", data=irl_sub)
plt.xlabel('New Cases')
plt.ylabel('Total Cases')
plt.title('Scatterplot for the Association between New Cases Rate and Total Cases')
plt.show()

#scatter plot for NEW_TESTS : TOTAL_CASES
scat2 = seaborn.regplot(x="NEW_TESTS", y="TOTAL_CASES", data=irl_sub)
plt.xlabel('New Tests')
plt.ylabel('Total Cases')
plt.title('Scatterplot for the Association between New Tests and Total Cases')
plt.show()

#scatter plot for NEW_TESTS : NEW_CASES
scat3 = seaborn.regplot(x="NEW_TESTS", y="NEW_CASES", data=irl_sub)
plt.xlabel('Tests')
plt.ylabel('Cases')
plt.title('Scatterplot for the Association between New Tests and New Cases')
plt.show()


## LINE GRAPH FOR IRL <AND> DEU for INFECTED_DEATH_RATIO_TOTAL 
x_labels = irl_sub['DATE']
y_cases_tot_irl = irl_sub['INFECTED_DEATH_RATIO_TOTAL']
y_cases_tot_deu = deu_sub['INFECTED_DEATH_RATIO_TOTAL']

plt.plot(x_labels, y_cases_tot_irl,color='green')
plt.plot(x_labels, y_cases_tot_deu,color='black')

plt.title('Total Cases:Total Deaths Ratio (IRL & DEU)')
plt.xlabel('DATE')
plt.ylabel('C:D Ratio')


green_patch = mpatches.Patch(color='green', label='IRELAND')
black_patch = mpatches.Patch(color='black', label='GERMANY')

plt.legend(handles=[green_patch,black_patch])
plt.grid(True)
plt.show()

#FOCUS ON IRELAND 
##COUNTY BY COUNTY
#Infection - Death Rate Comp county b county

## Leinster
x_labels = leinster_sub['DATE']
y_ld = leinster_sub['LONGFORD']
y_wh = leinster_sub['WESTMEATH']
y_mh = leinster_sub['MEATH']
y_lh = leinster_sub['LOUTH']
y_d = leinster_sub['DUBLIN']
y_ww = leinster_sub['WICKLOW']
y_ke = leinster_sub['KILDARE']
y_oy = leinster_sub['OFFALY']
y_ls = leinster_sub['LAOIS']
y_kk = leinster_sub['KILKENNY']
y_cw = leinster_sub['CARLOW']
y_wx = leinster_sub['WEXFORD']


plt.plot(x_labels, y_ld, color='brown')
plt.plot(x_labels, y_wh, color='green')
plt.plot(x_labels, y_mh, color='maroon')
plt.plot(x_labels, y_lh, color='blue')
plt.plot(x_labels, y_d, color='indigo')

plt.plot(x_labels, y_ww, color='gold')
plt.plot(x_labels, y_ke, color='teal')
plt.plot(x_labels, y_oy, color='red')
plt.plot(x_labels, y_ls, color='black')
plt.plot(x_labels, y_kk, color='magenta')

plt.plot(x_labels, y_cw, color='crimson')
plt.plot(x_labels, y_wx, color='pink')

plt.title('Total Cases LEINSTER (IRL)')
plt.xlabel('DATE')
plt.ylabel('CASES')


brown = mpatches.Patch(color='brown', label='LONGFORD')
green = mpatches.Patch(color='green', label='WESTMEATH')
maroon = mpatches.Patch(color='maroon', label='MEATH')

blue = mpatches.Patch(color='blue', label='LOUTH')
indigo = mpatches.Patch(color='indigo', label='DUBLIN')
gold = mpatches.Patch(color='gold', label='WICKLOW')

teal = mpatches.Patch(color='teal', label='KILDARE')
red = mpatches.Patch(color='red', label='OFFALY')
black = mpatches.Patch(color='black', label='LAOIS')

magenta = mpatches.Patch(color='magenta', label='KILKENNY')
crimson = mpatches.Patch(color='crimson', label='CARLOW')
pink = mpatches.Patch(color='pink', label='WEXFORD')

plt.legend(handles=[brown,green,maroon,blue,indigo,gold,teal,red,black,magenta,crimson,pink])
plt.grid(True)
plt.show()

## Ulster
x_labels = ulster_sub['DATE']
y_dl = ulster_sub['DONEGAL']
y_cn = ulster_sub['CAVAN']
y_mn = ulster_sub['MONAGHAN']


plt.plot(x_labels, y_dl, color='brown')
plt.plot(x_labels, y_cn, color='green')
plt.plot(x_labels, y_mn, color='maroon')

plt.title('Total Cases ULSTER (IRL)')
plt.xlabel('DATE')
plt.ylabel('CASES')


brown = mpatches.Patch(color='brown', label='DONEGAL')
green = mpatches.Patch(color='green', label='WESTMEATH')
maroon = mpatches.Patch(color='maroon', label='CAVAN')

plt.legend(handles=[brown,green,maroon,blue])
plt.grid(True)
plt.show()


## Connacht
x_labels = connacht_sub['DATE']
y_lm = connacht_sub['LEITRIM']
y_so = connacht_sub['SLIGO']
y_mo = connacht_sub['MAYO']
y_rn = connacht_sub['ROSCOMMON']
y_gy = connacht_sub['GALWAY']

plt.plot(x_labels, y_lm, color='brown')
plt.plot(x_labels, y_so, color='green')
plt.plot(x_labels, y_mo, color='maroon')
plt.plot(x_labels, y_rn, color='blue')
plt.plot(x_labels, y_gy, color='indigo')

plt.title('Total Cases CONNACHT (IRL)')
plt.xlabel('DATE')
plt.ylabel('CASES')

brown = mpatches.Patch(color='brown', label='LEITRIM')
green = mpatches.Patch(color='green', label='SLIGO')
maroon = mpatches.Patch(color='maroon', label='MAYO')
blue = mpatches.Patch(color='blue', label='ROSCOMMON')
indigo = mpatches.Patch(color='indigo', label='GALWAY')

plt.legend(handles=[brown,green,maroon,blue,indigo])
plt.grid(True)
plt.show()

## MUNSTER
x_labels = munster_sub['DATE']
y_wd = munster_sub['WATERFORD']
y_t = munster_sub['TIPPERARY']
y_ce = munster_sub['CLARE']
y_lk = munster_sub['LIMERICK']
y_ck = munster_sub['CORK']
y_ky = munster_sub['KERRY']

plt.plot(x_labels, y_wd, color='brown')
plt.plot(x_labels, y_t, color='green')
plt.plot(x_labels, y_ce, color='maroon')
plt.plot(x_labels, y_lk, color='blue')
plt.plot(x_labels, y_ck, color='indigo')
plt.plot(x_labels, y_ky, color='gold')

plt.title('Total Cases MUNSTER (IRL)')
plt.xlabel('DATE')
plt.ylabel('CASES')

brown = mpatches.Patch(color='brown', label='WATERFORD')
green = mpatches.Patch(color='green', label='TIPPERARY')
maroon = mpatches.Patch(color='maroon', label='CLARE')
blue = mpatches.Patch(color='blue', label='LIMERICK')
indigo = mpatches.Patch(color='indigo', label='CORK')
gold = mpatches.Patch(color='gold', label='KERRY')

plt.legend(handles=[brown,green,maroon,blue,indigo,gold])
plt.grid(True)
plt.show()


          #
          ##
#############
##############     QUESTION 2
#############
          ##
          #

#https://www.idc.com/promo/pcdforecast
#https://www.canalys.com/newsroom/canalys-global-pc-market-forecasts-2021
#https://www.semiconductors.org/chipmakers-are-ramping-up-production-to-address-semiconductor-shortage-heres-why-that-takes-time/


## LINE GRAPH FOR VACCINATED:CASES RATE IRL


x_labels = irl_sub['DATE']
y1_labels = pcd_sub['NOTEBOOK']
y2_labels = irl_sub['NEW_CASES']

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

spl = make_interp_spline(x_labels, y1_labels, k=3)
y_smooth = spl(x_labels)

curve1 = ax1.plot(x_labels, y_smooth, label = 'C1', color = 'red')
curve2 = ax2.plot(x_labels, y2_labels, label = 'C2', color = 'blue')

ax1.set_label('C1')
ax2.set_label('C2')

plt.title('IRL New Cases : PCD Shipments for Notebooks')

plt.plot()
plt.show()



#QUESTION 2??? Mental Health
#https://pubmed.ncbi.nlm.nih.gov/32716520/ 
#https://www.independent.ie/life/health-wellbeing/mental-health/demand-for-mental-health-and-suicide-prevention-services-soars-during-covid-19-pandemic-40279035.html
#https://www.irishtimes.com/news/health/ireland-facing-a-tsunami-of-mental-health-problems-1.4273850
#https://onlinelibrary.wiley.com/doi/10.1111/acps.13219



#QUESTION 3?? Smoking impact on deathrates
#https://www.hse.ie/eng/about/who/tobaccocontrol/research/smoking-in-ireland-2020.pdf




