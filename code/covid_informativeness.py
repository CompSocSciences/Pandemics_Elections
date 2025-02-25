import warnings 
warnings.filterwarnings("ignore")
import os
import gc
import time
import math
import patsy
import random
import imageio
import graphviz
import datetime
# from dfply import *
import numpy as np
from numpy import mean,std,absolute
import pandas as pd
import seaborn as sns
from sklearn import tree
# import missingno as msno
# from pingouin import mwu
from pingouin import ancova
from patsy import dmatrices
# from ExKMC.Tree import Tree
import statsmodels.api as sm
from numpy import linalg as LA
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.linear_model import LogisticRegressionCV,LinearRegression,ElasticNet,ElasticNetCV,Ridge, Lasso,RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error,r2_score,jaccard_score,confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score,RepeatedKFold
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import PolynomialFeatures,LabelEncoder,normalize
from scipy.stats import ks_2samp,pearsonr,mannwhitneyu
from sklearn.cluster import KMeans
from scipy.spatial import distance
import scipy.cluster.hierarchy as shc
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist
# from imblearn.over_sampling import SMOTE
from scipy.sparse import csc_matrix, find
from scipy.special import gammaln as lgamma
from scipy.signal import find_peaks, peak_widths
from statsmodels.genmod.families import Binomial
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.model_selection import train_test_split 
from sklearn.datasets import make_classification
from statsmodels.base.model import GenericLikelihoodModel
# from imblearn.over_sampling import ADASYN, SMOTENC, BorderlineSMOTE
from scipy.stats import studentized_range

os.chdir("C:/Users/neilh/Documents/Research/Covid/political")
from utils import median_distance,kernel_regression_fit_and_predict
from data_generation_methods import plot_kr_scatter_county

county_df6 = pd.read_pickle("data/county_df6.pkl")
covid = pd.read_csv("data/covid.csv")
covid = covid.iloc[:,1:12]
covidSt = pd.read_csv("data/covidSt.csv")
covidSt['cases_n'] = covidSt.cases / covidSt['pop'] * 10000000
covidSt['deaths_n'] = covidSt.deaths / covidSt['pop'] * 10000000

wi = county_df6.loc[county_df6.state_po=='WI',['state_po','CountyName','countyFIPS','PopulationEstimate2018','R_votes_2016_pct','R_votes_2020_pct','R_votes_diff_diff','D_votes_diff_diff','no_peaks_c','no_peaks_d','nearest_peak_c','nearest_peak_d','nearest_area_c','nearest_area_d']]
republican = county_df6.loc[county_df6.R_votes_2016_pct>0.5,['state_po','CountyName','countyFIPS','PopulationEstimate2018','R_votes_2016_pct','R_votes_2020_pct','R_votes_diff_diff','D_votes_2016_pct','D_votes_2020_pct','D_votes_diff_diff','no_peaks_c','no_peaks_d','nearest_peak_c','nearest_peak_d','nearest_area_c','nearest_area_d']]
democrat = county_df6.loc[county_df6.D_votes_2016_pct>0.5,['state_po','CountyName','countyFIPS','PopulationEstimate2018','D_votes_2016_pct','D_votes_2020_pct','D_votes_diff_diff','R_votes_2016_pct','R_votes_2020_pct','R_votes_diff_diff','no_peaks_c','no_peaks_d','nearest_peak_c','nearest_peak_d','nearest_area_c','nearest_area_d']]
# temp = county_df6.loc[:,['state_po','CountyName','countyFIPS','PopulationEstimate2018','R_votes_2016_pct','R_votes_2020_pct','D_votes_2016_pct','D_votes_2020_pct','R_votes_diff_diff','D_votes_diff_diff','no_peaks_c','no_peaks_d','nearest_peak_c','nearest_peak_d','nearest_area_c','nearest_area_d']]
# peak for Henry GA
min(covid.loc[covid.fips==13151,'date']) #add the next line
county_df6.loc[county_df6.index==353,'nearest_peak_c']
# peak for Henry GA
min(covid.loc[covid.fips==13151,'date']) + county_df6.loc[county_df6.index==353,'nearest_peak_c']

temp1=county_df6.loc[county_df6.index==353,:] # Henry GA
temp1=county_df6.loc[county_df6.index==361,:] #Robeson NC

plot_kr_scatter_county(['Robeson_NC','Henry_GA'],type='cases')
plot_kr_scatter_county(['Edwards_TX','Edwards_TX'],type='cases')
# plot_kr_scatter_county(['Webb_TX','Starr_TX'],type='deaths')


covidSt_df1= pd.read_pickle("data/covidSt_df1.pkl")
emotional_health2= pd.read_pickle("data/emotional_health2.pkl")
state_region_xwalk = pd.read_pickle('data/state_region_xwalk.pkl')
# county_df5 = pd.read_pickle("data/county_df5.pkl")
# county_df6 = pd.read_pickle("data/county_df6.pkl")
senate_2016 = pd.read_csv('data/2016-precinct-senate-NH.csv', sep=',', lineterminator='\r')
senate_2020 = pd.read_csv('data/2020-SENATE-precinct-general-NH.csv', sep=',')

total_votes_Senate_2016 = {}
D_votes_Senate_2016 = {}
R_votes_Senate_2016 = {}
I_votes_Senate_2016 = {}
total_votes_Senate_2020 = {}
D_votes_Senate_2020 = {}
R_votes_Senate_2020 = {}
I_votes_Senate_2020 = {}

for fip in senate_2016.county_fips.unique().tolist():
  if fip in county_df6.countyFIPS.tolist():
    temp = senate_2016[senate_2016.county_fips==fip]
    total_votes_Senate_2016[fip] = temp.loc[:,'votes'].sum()
    D_votes_Senate_2016[fip] = temp.loc[temp.party=='democratic','votes'].sum()
    R_votes_Senate_2016[fip] = temp.loc[temp.party=='republican','votes'].sum()
    I_votes_Senate_2016[fip] = temp.loc[pd.isna(temp.party),'votes'].sum()

for fip in senate_2020.county_fips.unique().tolist():
  if fip in county_df6.countyFIPS.tolist():
    temp = senate_2020[senate_2020.county_fips==fip]
    total_votes_Senate_2020[fip] = temp.loc[:,'votes'].sum()
    D_votes_Senate_2020[fip] = temp.loc[temp.party_simplified=='DEMOCRAT','votes'].sum()
    R_votes_Senate_2020[fip] = temp.loc[temp.party_simplified=='REPUBLICAN','votes'].sum()
    I_votes_Senate_2020[fip] = temp.loc[temp['party_simplified'].isnull(),'votes'].sum()

# D_votes_Senate_2016 = D_votes_Senate_2016.iloc[: , 1:]
# D_votes_Senate_2020 = D_votes_Senate_2020.iloc[: , 1:]
# R_votes_Senate_2016 = R_votes_Senate_2016.iloc[: , 1:]
# R_votes_Senate_2020 = R_votes_Senate_2020.iloc[: , 1:]
# I_votes_Senate_2016 = I_votes_Senate_2016.iloc[: , 1:]
# I_votes_Senate_2020 = I_votes_Senate_2020.iloc[: , 1:]
# total_votes_Senate_2016 = total_votes_Senate_2016.iloc[: , 1:]
# total_votes_Senate_2020 = total_votes_Senate_2020.iloc[: , 1:]

D_votes_Senate_2016 = pd.DataFrame(D_votes_Senate_2016, index=[0]).T
D_votes_Senate_2020 = pd.DataFrame(D_votes_Senate_2020, index=[0]).T
total_votes_Senate_2016 = pd.DataFrame(total_votes_Senate_2016, index=[0]).T
total_votes_Senate_2020 = pd.DataFrame(total_votes_Senate_2020, index=[0]).T

# D_votes_Senate_2016 = pd.DataFrame(D_votes_Senate_2016).T
# D_votes_Senate_2020 = pd.DataFrame(D_votes_Senate_2020).T
# total_votes_Senate_2016 = pd.DataFrame(total_votes_Senate_2016).T
# total_votes_Senate_2020 = pd.DataFrame(total_votes_Senate_2020).T
D_votes_Senate_2016.rename({0:'D_votes_Senate_2016'}, axis=1, inplace=True)
D_votes_Senate_2020.rename({0:'D_votes_Senate_2020'}, axis=1, inplace=True)
total_votes_Senate_2016.rename({0:'total_votes_Senate_2016'}, axis=1, inplace=True)
total_votes_Senate_2020.rename({0:'total_votes_Senate_2020'}, axis=1, inplace=True)

R_votes_Senate_2016 = pd.DataFrame(R_votes_Senate_2016, index=[0]).T
R_votes_Senate_2020 = pd.DataFrame(R_votes_Senate_2020, index=[0]).T
# R_votes_Senate_2016 = pd.DataFrame(R_votes_Senate_2016).T
# R_votes_Senate_2020 = pd.DataFrame(R_votes_Senate_2020).T
R_votes_Senate_2016.rename({0:'R_votes_Senate_2016'}, axis=1, inplace=True)
R_votes_Senate_2020.rename({0:'R_votes_Senate_2020'}, axis=1, inplace=True)

I_votes_Senate_2016 = pd.DataFrame(I_votes_Senate_2016, index=[0]).T
I_votes_Senate_2020 = pd.DataFrame(I_votes_Senate_2020, index=[0]).T
# I_votes_Senate_2016 = pd.DataFrame(I_votes_Senate_2016).T
# I_votes_Senate_2020 = pd.DataFrame(I_votes_Senate_2020).T
I_votes_Senate_2016.rename({0:'I_votes_Senate_2016'}, axis=1, inplace=True)
I_votes_Senate_2020.rename({0:'I_votes_Senate_2020'}, axis=1, inplace=True)


county_senate1 = county_df6.merge(total_votes_Senate_2016,how='inner',left_on='countyFIPS',right_index=True)                                  
county_senate1 = county_senate1.merge(D_votes_Senate_2016,how='inner',left_on='countyFIPS',right_index=True)
county_senate1 = county_senate1.merge(R_votes_Senate_2016,how='inner',left_on='countyFIPS',right_index=True)
county_senate1 = county_senate1.merge(I_votes_Senate_2016,how='inner',left_on='countyFIPS',right_index=True)
county_senate1 = county_senate1.merge(total_votes_Senate_2020,how='inner',left_on='countyFIPS',right_index=True)
county_senate1 = county_senate1.merge(D_votes_Senate_2020,how='inner',left_on='countyFIPS',right_index=True)
county_senate1 = county_senate1.merge(R_votes_Senate_2020,how='inner',left_on='countyFIPS',right_index=True)
county_senate1 = county_senate1.merge(I_votes_Senate_2020,how='inner',left_on='countyFIPS',right_index=True)
county_senate1.loc[:,'D_votes_Senate_diff']     = county_senate1.D_votes_Senate_2020    - county_senate1.D_votes_Senate_2016
county_senate1.loc[:,'D_votes_Senate_2016_pct'] = county_senate1.D_votes_Senate_2016    / county_senate1.total_votes_Senate_2016
county_senate1.loc[:,'D_votes_Senate_2020_pct'] = county_senate1.D_votes_Senate_2020    / county_senate1.total_votes_Senate_2020
county_senate1.loc[:,'D_votes_Senate_diff_pct'] = county_senate1.D_votes_Senate_2020_pct- county_senate1.D_votes_Senate_2016_pct

county_senate1.loc[:,'R_votes_Senate_diff']     = county_senate1.R_votes_Senate_2020    - county_senate1.R_votes_Senate_2016
county_senate1.loc[:,'R_votes_Senate_2016_pct'] = county_senate1.R_votes_Senate_2016    / county_senate1.total_votes_Senate_2016
county_senate1.loc[:,'R_votes_Senate_2020_pct'] = county_senate1.R_votes_Senate_2020    / county_senate1.total_votes_Senate_2020
county_senate1.loc[:,'R_votes_Senate_diff_pct'] = county_senate1.R_votes_Senate_2020_pct- county_senate1.R_votes_Senate_2016_pct

county_senate1['R_votes_Senate_diff_diff'] = county_senate1.R_votes_Senate_diff_pct - county_senate1.D_votes_Senate_diff_pct
county_senate1['D_votes_Senate_diff_diff'] = county_senate1.D_votes_Senate_diff_pct - county_senate1.R_votes_Senate_diff_pct

county_senate1.to_pickle("data/county_senate1.pkl")
# county_senate1.to_pickle("county_senate2.pkl")



republican_senate = county_senate1.loc[county_senate1.R_votes_Senate_2016_pct>0.5,['state_po','CountyName','countyFIPS','PopulationEstimate2018','R_votes_Senate_2016_pct','R_votes_Senate_2020_pct','R_votes_Senate_diff_pct','D_votes_Senate_2016_pct','D_votes_Senate_2020_pct','D_votes_Senate_diff_diff','no_peaks_c','no_peaks_d','nearest_peak_c','nearest_peak_d','nearest_area_c','nearest_area_d']]
democrat_senate = county_senate1.loc[county_senate1.D_votes_2016_pct>0.5,['state_po','CountyName','countyFIPS','PopulationEstimate2018','D_votes_Senate_2016_pct','D_votes_Senate_2020_pct','D_votes_Senate_diff_pct','R_votes_Senate_2016_pct','R_votes_Senate_2020_pct','R_votes_Senate_diff_diff','no_peaks_c','no_peaks_d','nearest_peak_c','nearest_peak_d','nearest_area_c','nearest_area_d']]
# temp = county_df6.loc[:,['state_po','CountyName','countyFIPS','PopulationEstimate2018','R_votes_2016_pct','R_votes_2020_pct','D_votes_2016_pct','D_votes_2020_pct','R_votes_diff_diff','D_votes_diff_diff','no_peaks_c','no_peaks_d','nearest_peak_c','nearest_peak_d','nearest_area_c','nearest_area_d']]
# peak for Henry GA
min(covid.loc[covid.fips==13151,'date']) #add the next line
county_df6.loc[county_df6.index==353,'nearest_peak_c']
# peak for Henry GA
min(covid.loc[covid.fips==13151,'date']) + county_df6.loc[county_df6.index==353,'nearest_peak_c']

temp1=county_df6.loc[county_df6.index==353,:] # Henry GA
temp1=county_df6.loc[county_df6.index==361,:] #Robeson NC

# plot_kr_scatter_county(['Lincoln_AR','Linn_IA'],type='cases')

## troubleshoot later: "ValueError: Multi-dimensional indexing (e.g. `obj[:, None]`) is no longer supported. Convert to a numpy array before indexing instead."
plot_kr_scatter_county(['Yuma_CO','Yuma_CO'],type='cases',covid=covid)
plot_kr_scatter_county(['Edwards_TX','Edwards_TX'],type='cases')

plot_kr_scatter_uk(['E07000090','E07000229'],type='cases')


## Moderate Republican  #mixed: strong death area and death peak, opposite case area
temp = county_senate1[county_senate1.R_votes_Senate_2016_pct < 0.7]
temp = temp[temp.R_votes_Senate_2016_pct >= 0.5]

## Strong Republican  #strong death area
temp = county_senate1[county_senate1.R_votes_Senate_2016_pct > 0.7]

## Moderate Democrat  #mixed: strong case area, opposite death area, death peak
temp = county_senate1[county_senate1.D_votes_Senate_2016_pct < 0.7]
temp = temp[temp.D_votes_Senate_2016_pct >= 0.5]

## Strong Democrat  #Not enough data
temp = county_senate1[county_senate1.D_votes_Senate_2016_pct > 0.7]



####################################################################
####################### JOP paper submission #######################

############################## Senate     ##########################
############################## Regression ##########################


county_senate1 = pd.read_pickle("data/county_senate1.pkl")

temp = county_senate1[county_senate1.D_votes_Senate_2016_pct > county_senate1.R_votes_Senate_2016_pct]
Y, X1 = dmatrices('D_votes_Senate_diff_pct ~ state_po + rural_urban +\
medicare + density + black_pct + nearest_peak_c + nearest_peak_d + \
        male_pct + hispanic_pct + pop_20_29_pct+ pop_30_39_pct+ pop_65_up_pct + \
            HeartDiseaseMortality+ RespMortalityRate2014 + SVIPercentile', \
    temp, return_type="dataframe")


# Y, X1 = dmatrices('R_votes_Senate_diff_pct ~ state_po + \
#     medicare + density + black_pct + nearest_peak_c + nearest_peak_d +\
#             male_pct + hispanic_pct + pop_20_29_pct+ pop_30_39_pct+ pop_65_up_pct + \
#                 HeartDiseaseMortality+ RespMortalityRate2014 + SVIPercentile', \
#         temp, return_type="dataframe")

temp = county_senate1[county_senate1.R_votes_Senate_2016_pct > county_senate1.D_votes_Senate_2016_pct]
Y, X1 = dmatrices('R_votes_Senate_diff_pct ~ state_po + rural_urban +\
medicare + density + black_pct + nearest_peak_c + nearest_peak_d +\
        male_pct + hispanic_pct + pop_20_29_pct+ pop_30_39_pct+ pop_65_up_pct + \
            HeartDiseaseMortality + RespMortalityRate2014 + SVIPercentile', \
    temp, return_type="dataframe")

    
Y = np.ravel(Y)
model1 = sm.OLS(Y,X1.astype(float))
results1 = model1.fit()

coefs1 = pd.DataFrame({
    'coef': results1.params.values,
    'stderr': results1.bse,
    'pvalue': results1.pvalues
}).sort_values(by='pvalue', ascending=True)

coefs1.loc['nearest_peak_c',['coef','pvalue','stderr']]
coefs1.loc['nearest_peak_d',['coef','pvalue','stderr']]
# coefs1.loc['nearest_area_d',['coef','pvalue','stderr']]
# coefs1.loc['nearest_area_c',['coef','pvalue','stderr']]
# coefs1.loc['no_peaks_c',['coef','pvalue','stderr']]


####################################################################
################################# JOP end ##########################
####################################################################


####
county_df6['R_votes_diff_diff'] = county_df6.R_votes_diff_pct - county_df6.D_votes_diff_pct
county_df6['D_votes_diff_diff'] = county_df6.D_votes_diff_pct - county_df6.R_votes_diff_pct


## Moderate Republican
temp = county_senate1[county_senate1.R_votes_Senate_2016_pct < 0.7]
temp = temp[temp.R_votes_Senate_2016_pct > 0.5]

temp = county_senate1[county_senate1.R_votes_Senate_2016_pct > 0.5]

temp = county_df6[county_df6.D_votes_2016_pct < 0.7]
temp = temp[temp.D_votes_2016_pct > 0.5]

# Y, X6 = dmatrices('D_votes_diff_diff~ state_po + hispanic_pct + male_pct +\
#                   black_pct + pop_20_29_pct+ pop_30_39_pct+ pop_65_up_pct + \
#                       tot_pop + MedianAge2010 + density + \
#                           nearest_peak_c + nearest_peak_d + nearest_area_d + \
#                               nearest_area_c + no_peaks_c + no_peaks_d +\
#                                   HeartDiseaseMortality+ RespMortalityRate2014'\
#                                       ,temp, return_type="dataframe")
Y, X6 = dmatrices('R_votes_diff_diff ~ state_po + tot_pop +\
MedianAge2010 + pop_65_up_pct + medicare + density + rural_urban + black_pct +\
    nearest_peak_c + nearest_area_c + nearest_peak_d + nearest_area_d +\
        male_pct + hispanic_pct + tot_pop + pop_20_29_pct+ pop_30_39_pct+ pop_65_up_pct + \
            no_peaks_c + no_peaks_d + rural_urban + Smokers_Percentage + SVIPercentile +\
                StrokeMortality + HeartDiseaseMortality+ RespMortalityRate2014', \
                                      temp, return_type="dataframe")
Y = np.ravel(Y)

model6 = sm.OLS(Y,X6.astype(float))
results6 = model6.fit()

coefs6 = pd.DataFrame({
    'coef': results6.params.values,
    'stderr': results6.bse,
    'pvalue': results6.pvalues
}).sort_values(by='pvalue', ascending=True)
coefs6.loc['nearest_area_d',:]
# coefs6.loc['nearest_area_c',:]
# coefs6.loc['nearest_peak_d',:]
# coefs6.loc['nearest_peak_c',:]
# coefs6.loc['no_peaks_c',:]


######################################################
### Presidential election

county_df6.to_csv('data/county_df6.csv')


county_df6['R_votes_diff_diff'] = county_df6.R_votes_diff_pct - county_df6.D_votes_diff_pct
county_df6['D_votes_diff_diff'] = county_df6.D_votes_diff_pct - county_df6.R_votes_diff_pct
# county_df6.to_pickle('data/county_df6.pkl')

# Moderate Democrats
temp = county_df6[county_df6.D_votes_2016_pct <= 0.7] #works but 13%
temp = temp[temp.D_votes_2016_pct > 0.5]

# Strong Democrats (opposite)
temp = county_df6[county_df6.D_votes_2016_pct > 0.7]  #(opposite)


# Moderate Republicans
temp = county_df6[county_df6.R_votes_2016_pct <= 0.7]  #opposite death area death peak
temp = temp[temp.R_votes_2016_pct > 0.5]

# Strong Republicans
temp = county_df6[county_df6.R_votes_2016_pct > 0.7]  #works strong death area



###########################################################################
############### for Journal of Politics JOP submission ####################


################################ Viz ###############################
reps = county_df6.loc[county_df6.R_votes_2016_pct > county_df6.D_votes_2016_pct,\
                      ['R_votes_diff_pct','CensusRegionName','rural_urban']  ]
dems = county_df6.loc[county_df6.D_votes_2016_pct > county_df6.R_votes_2016_pct,\
                      ['D_votes_diff_pct','CensusRegionName','rural_urban']  ]
f, (ax1, ax2,ax3,ax4) = plt.subplots(1,4, sharey=True)
sns.boxplot(x=dems["CensusRegionName"], y=dems["D_votes_diff_pct"],\
            palette="Reds", ax=ax1).set_title('Democratic Counties', loc = "left")
ax1.axhline(y=0, c='black') 
ax1.set(xlabel = "")
ax1.tick_params(axis='x', rotation=40)
ax1.set(ylabel = "Vote Share Change: 2020 - 2016")

sns.boxplot(x=dems["rural_urban"], y=dems["D_votes_diff_pct"],\
            palette="Reds", ax=ax2)
ax2.axhline(y=0, c='black') 
ax2.set(xlabel = "")
ax2.set(ylabel = "")
ax2.tick_params(axis='x', rotation=0)

sns.boxplot(x=reps["CensusRegionName"], y=reps["R_votes_diff_pct"],\
                palette="Blues", ax=ax3).set_title('Republican Counties', loc = "left")
ax3.axhline(y=0, c='black') 
ax3.set(xlabel = "")
ax3.tick_params(axis='x', rotation=40)
ax3.set(ylabel = "")

sns.boxplot(x=reps["rural_urban"], y=reps["R_votes_diff_pct"],\
                palette="Blues", ax=ax4)
ax4.axhline(y=0, c='black') 
ax4.set(xlabel = "")
ax4.tick_params(axis='x', rotation=0)
ax4.set(ylabel = "")



#########################################################################
##############################  US President ############################
##############################  Regression  #############################
#Y variable: 
#1. R_votes_diff_pct
#2. D_votes_diff_pct
#3. R_votes_diff_diff
#4. D_votes_diff_diff
# Y, X = dmatrices('D_votes_diff_pct ~ state_po + \
# MedianAge2010 + pop_65_up_pct + medicare + black_pct +\
#     nearest_peak_c + nearest_area_c + nearest_peak_d + nearest_area_d +\
#         male_pct + hispanic_pct + tot_pop + pop_20_29_pct+ pop_30_39_pct+ pop_65_up_pct + \
#             no_peaks_c + no_peaks_d + rural_urban + SVIPercentile +\
#                 RespMortalityRate2014', \
# temp, return_type="dataframe")#sig. covariates: R_votes_2016_pct(-),D_votes_2016_pct(-)

# All Democrats
temp2 = county_df6[county_df6.D_votes_2016_pct > county_df6.R_votes_2016_pct]  

Y2, X2 = dmatrices('D_votes_diff_pct ~ state_po + medicare + density + black_pct + \
                 nearest_peak_c + nearest_peak_d + male_pct + hispanic_pct + pop_20_29_pct+ \
                     pop_30_39_pct+ pop_65_up_pct + HeartDiseaseMortality + RespMortalityRate2014+\
                         SVIPercentile+ rural_urban', temp2, return_type="dataframe")

Y2 = np.ravel(Y2)
model2 = sm.OLS(Y2,X2.astype(float))
results2 = model2.fit()
coefs2 = pd.DataFrame({
    'coef': results2.params.values,
    'stderr': results2.bse,
    'pvalue': results2.pvalues
}).sort_values(by='pvalue', ascending=True)

coefs2.loc['nearest_peak_c',:]
coefs2.loc['nearest_peak_d',:]


# All Republicans
# temp = county_df6[county_df6.R_votes_2016_pct > 0.5]  #opposite strong death peak
temp = county_df6[county_df6.R_votes_2016_pct > county_df6.D_votes_2016_pct]  
Y, X = dmatrices('R_votes_diff_pct ~ state_po + tot_pop + medicare + density + black_pct +\
    nearest_peak_c + nearest_peak_d +male_pct + hispanic_pct + pop_20_29_pct+ pop_30_39_pct+ \
        pop_65_up_pct + HeartDiseaseMortality+ RespMortalityRate2014 +SVIPercentile +\
            rural_urban', temp, return_type="dataframe")
   
Y = np.ravel(Y)
model1 = sm.OLS(Y,X.astype(float))
results1 = model1.fit()
coefs1 = pd.DataFrame({
    'coef': results1.params.values,
    'stderr': results1.bse,
    'pvalue': results1.pvalues
}).sort_values(by='pvalue', ascending=True)
coefs1.loc['nearest_peak_c',:]
coefs1.loc['nearest_peak_d',:]

    



# HOUSE
house = pd.read_csv("data/1976-2020-house.tab",sep='\t')
house = house[house.year == 2018]
state_house = pd.read_pickle("data/house.pkl")



# UK
uk = pd.read_csv("data/ltla_2022-04-08.csv")
uk = uk.loc[:,['areaCode','date','newCasesByPublishDate','newDeaths28DaysByPublishDate']]
uk = uk.fillna(0)
uk.rename({'newCasesByPublishDate':'cases','newDeaths28DaysByPublishDate':'deaths'}, axis=1, inplace=True)
cutoff = pd.to_datetime('20210506', format='%Y%m%d', errors='ignore') # 1 day before election date 5-7-2021
uk['date'] = pd.to_datetime(uk['date'], format='%m/%d/%Y', errors='ignore')
uk = uk[uk.date < cutoff]

max_yaxis = 0
max_xaxis = 0
wl = 7

for areaCode in list(uk.areaCode.unique()):
    covid_plot = uk[uk.areaCode == areaCode]    
    covid_plot["daily_d"] = covid_plot.deaths.diff()    
    covid_plot["daily_c"] = covid_plot.cases.diff()
    covid_plot.loc[covid_plot.daily_d<0,'daily_d'] = 0
    covid_plot.loc[covid_plot.daily_c<0,'daily_c'] = 0
    covid_plot["ma_d"] = np.rint(covid_plot.daily_d.rolling(window=wl).mean())
    covid_plot["ma_c"] = np.rint(covid_plot.daily_c.rolling(window=wl).mean())
    covid_plot = covid_plot.iloc[wl:]
    if (covid_plot.shape[0]>0):
      max_of_d_and_c = np.max([covid_plot.ma_d,covid_plot.ma_c])
      if max_of_d_and_c*1.1 > max_yaxis:
          max_yaxis = max_of_d_and_c*1.1
      if covid_plot.index.shape[0]+wl > max_xaxis:
          max_xaxis = covid_plot.index.shape[0]+wl

print(max_xaxis, max_yaxis)

# Output from above
max_xaxis = 430
max_yaxis = 507
wl = 7

peaks_arr = {}
widths_arr = {}
heights_arr = {}
areas_arr = {}
areas_KR_arr = {}
no_peaks_arr = {}

beta = 0.3
max_no_height =8

for fip in list(uk.areaCode.unique()):
    covid_plot = uk[uk.areaCode == fip]
    pad_zeros = max_xaxis - covid_plot.shape[0]
    zero_df = pd.DataFrame(0, index=np.arange(pad_zeros), columns=covid_plot.columns)
    covid_plot = pd.concat([zero_df,covid_plot])
    # covid_plot["daily"] = covid_plot.cases.diff()        
    covid_plot["daily"] = covid_plot.deaths.diff()
    covid_plot.loc[covid_plot.daily<0,'daily'] = 0 
    covid_plot["ma"] = np.rint(covid_plot.daily.rolling(window=wl).mean())
    covid_plot = covid_plot.iloc[wl:,:]
    covid_plot = covid_plot.reset_index()
    covid_plot['periods'] = covid_plot.index        
    xTrain = covid_plot.periods.to_numpy()[:,np.newaxis]
    yTrain = covid_plot.ma
    h = beta*median_distance(xTrain)
    yHatk = kernel_regression_fit_and_predict(xTrain, yTrain, xTrain, h, beta)
    peaks, _ = find_peaks(yHatk)#, height=0.0005)
    if (np.argmax(yHatk) in peaks) == False and np.argmax(yHatk) > len(xTrain)-20:
        peaks = np.append(peaks,np.argmax(yHatk))      
    no_idx = len(peaks)
    pad_zeros = max_no_height - no_idx
    idx = yHatk[peaks].argsort()[::-1][:len(peaks)+1]    
    w, _, _, _= peak_widths(yHatk, peaks,rel_height=0.5)
    h = yHatk[peaks]    
    widths = w[idx] #reorder based on order of peaks    
    heights = h[idx]
    peaks_arr[fip] = np.pad(peaks[idx],pad_zeros,mode='constant')[pad_zeros:]
    widths_arr[fip] = np.pad(widths,pad_zeros,mode='constant')[pad_zeros:]
    heights_arr[fip] = np.pad(heights,pad_zeros,mode='constant')[pad_zeros:]
    no_peaks_arr[fip] = no_idx
    areas = []
    areas_kr = []
    for ind in idx:
      start_ind = max(0,peaks[ind]-int(np.round(w[ind]/2.0)))
      end_ind = min(len(yHatk),peaks[ind]+int(np.round(w[ind]/2.0)+1))
      yvalues_kr = yHatk[start_ind:end_ind]
      if min(yvalues_kr)<0:
        yvalues_kr = yvalues_kr - min(yvalues_kr)
      yvalues = covid_plot.daily[start_ind:end_ind]
      if min(yvalues)<0:
        yvalues = yvalues - min(yvalues)
      areas.append(sum(yvalues))
      areas_kr.append(sum(yvalues_kr))      
    areas_arr[fip] = np.pad(areas,pad_zeros,mode='constant')[pad_zeros:]
    areas_KR_arr[fip] = np.pad(areas_kr,pad_zeros,mode='constant')[pad_zeros:]    
    if len(heights_arr) % 100 == 0:
      gc.collect()
     

# Cases
peaks_c = pd.DataFrame.from_dict(peaks_arr).T
widths_c = pd.DataFrame.from_dict(widths_arr).T
heights_c = pd.DataFrame.from_dict(heights_arr).T
areas_c = pd.DataFrame.from_dict(areas_arr).T
areas_KR_c = pd.DataFrame.from_dict(areas_KR_arr).T
no_peaks_c   = pd.DataFrame(no_peaks_arr, index=[0]).T

peaks_c.rename({0:'peak1_c',1:'peak2_c',2:'peak3_c',3:'peak4_c',4:'peak5_c',5:'peak6_c',6:'peak7_c',7:'peak8_c'}, axis=1, inplace=True)
widths_c.rename({0:'width1_c',1:'width2_c',2:'width3_c',3:'width4_c',4:'width5_c',5:'width6_c',6:'width7_c',7:'width8_c'}, axis=1, inplace=True)
heights_c.rename({0:'height1_c',1:'height2_c',2:'height3_c',3:'height4_c',4:'height5_c',5:'height6_c',6:'height7_c',7:'height8_c'}, axis=1, inplace=True)
areas_c.rename({0:'area1_c',1:'area2_c',2:'area3_c',3:'area4_c',4:'area5_c',5:'area6_c',6:'area7_c',7:'area8_c'}, axis=1, inplace=True)
areas_KR_c.rename({0:'area1_KR_c',1:'area2_KR_c',2:'area3_KR_c',3:'area4_KR_c',4:'area5_KR_c',5:'area6_KR_c',6:'area7_KR_c',7:'area8_KR_c'}, axis=1, inplace=True)
no_peaks_c.rename({0:'no_peaks_c'}, axis=1, inplace=True)

# Death counts
peaks_d = pd.DataFrame.from_dict(peaks_arr).T
widths_d = pd.DataFrame.from_dict(widths_arr).T
heights_d = pd.DataFrame.from_dict(heights_arr).T
areas_d = pd.DataFrame.from_dict(areas_arr).T
areas_KR_d = pd.DataFrame.from_dict(areas_KR_arr).T
no_peaks_d = pd.DataFrame(no_peaks_arr, index=[0]).T

peaks_d.rename({0:'peak1_d',1:'peak2_d',2:'peak3_d',3:'peak4_d',4:'peak5_d',5:'peak6_d',6:'peak7_d',7:'peak8_d'}, axis=1, inplace=True)
widths_d.rename({0:'width1_d',1:'width2_d',2:'width3_d',3:'width4_d',4:'width5_d',5:'width6_d',6:'width7_d',7:'width8_d'}, axis=1, inplace=True)
heights_d.rename({0:'height1_d',1:'height2_d',2:'height3_d',3:'height4_d',4:'height5_d',5:'height6_d',6:'height7_d',7:'height8_d'}, axis=1, inplace=True)
areas_d.rename({0:'area1_d',1:'area2_d',2:'area3_d',3:'area4_d',4:'area5_d',5:'area6_d',6:'area7_d',7:'area8_d'}, axis=1, inplace=True)
areas_KR_d.rename({0:'area1_KR_d',1:'area2_KR_d',2:'area3_KR_d',3:'area4_KR_d',4:'area5_KR_d',5:'area6_KR_d',6:'area7_KR_d',7:'area8_KR_d'}, axis=1, inplace=True)
no_peaks_d.rename({0:'no_peaks_d'}, axis=1, inplace=True)


uk_df1 = peaks_c.merge(widths_c,how='inner',right_index=True,left_index=True)
uk_df1 = uk_df1.merge(heights_c,how='inner',right_index=True,left_index=True)
uk_df1 = uk_df1.merge(areas_c,how='inner',right_index=True,left_index=True)
uk_df1 = uk_df1.merge(areas_KR_c,how='inner',right_index=True,left_index=True)
# uk_df1 = uk_df1.merge(no_peaks_c,how='inner',right_index=True,left_index=True)

uk_df1 = uk_df1.merge(peaks_d,how='inner',right_index=True,left_index=True)
uk_df1 = uk_df1.merge(widths_d,how='inner',right_index=True,left_index=True)
uk_df1 = uk_df1.merge(heights_d,how='inner',right_index=True,left_index=True)
uk_df1 = uk_df1.merge(areas_d,how='inner',right_index=True,left_index=True)
uk_df1 = uk_df1.merge(areas_KR_d,how='inner',right_index=True,left_index=True)
# uk_df1 = uk_df1.merge(no_peaks_d,how='inner',right_index=True,left_index=True)

uk_df1.to_pickle('data/uk_df1.pkl')





## UK voting results
# uk_vote = pd.read_excel('data/CBP09228_detailed_results.xlsx', index_col=0,\
#                         sheet_name='Local election results')
uk_vote = pd.read_csv("data/CBP09228_detailed_results.csv",header=1)


# election_date = cutoff
min_date = np.min(uk.date) #datetime.datetime.strptime("03/02/2020","%m/%d/%Y").date()
election_days = (election_date-min_date).days

# case_peaks = ['peak1_c', 'peak2_c', 'peak3_c', 'peak4_c', 'peak5_c', 'peak6_c', 'peak7_c', 'peak8_c','no_peaks_c']
case_peaks = ['peak1_c', 'peak2_c', 'peak3_c', 'peak4_c', 'peak5_c', 'peak6_c', 'peak7_c', 'peak8_c']
case_areas = ['area1_c','area2_c','area3_c','area4_c','area5_c','area6_c','area7_c','area8_c',
              'area1_KR_c','area2_KR_c','area3_KR_c','area4_KR_c','area5_KR_c','area6_KR_c','area7_KR_c','area8_KR_c']
# death_peaks = ['peak1_d', 'peak2_d', 'peak3_d', 'peak4_d', 'peak5_d', 'peak6_d', 'peak7_d', 'peak8_d','no_peaks_d']
death_peaks = ['peak1_d', 'peak2_d', 'peak3_d', 'peak4_d', 'peak5_d', 'peak6_d', 'peak7_d', 'peak8_d']
death_areas = ['area1_d','area2_d','area3_d','area4_d','area5_d','area6_d','area7_d','area8_d',
              'area1_KR_d','area2_KR_d','area3_KR_d','area4_KR_d','area5_KR_d','area6_KR_d','area7_KR_d','area8_KR_d']

nearest_peak_c = {}
nearest_area_c = {}  
no_peaks_c = {}
nearest_peak_d = {}
nearest_area_d = {}
no_peaks_d = {}

for fip in uk_df1.index:
    no_peaks_c[fip] = 0
    no_peaks_d[fip] = 0
    for i,peak in enumerate(case_peaks):
      if (float(uk_df1.loc[uk_df1.index==fip,peak]) > 0):
        no_peaks_c[fip] = no_peaks_c[fip] + 1
        if fip in nearest_peak_c.keys():
            if float(uk_df1.loc[uk_df1.index==fip,peak]) > float(nearest_peak_c[fip]):
                nearest_peak_c[fip] = float(uk_df1.loc[uk_df1.index==fip,peak])
        else:
          nearest_peak_c[fip] = float(uk_df1.loc[uk_df1.index==fip,peak])
    for i,area in enumerate(case_areas):
      if (float(uk_df1.loc[uk_df1.index==fip,area]) > 0):
        if fip in nearest_area_c.keys():
            if float(uk_df1.loc[uk_df1.index==fip,area]) > float(nearest_area_c[fip]):
                nearest_area_c[fip] = float(uk_df1.loc[uk_df1.index==fip,area])
        else:                    
            nearest_area_c[fip] = float(uk_df1.loc[uk_df1.index==fip,area])
    for i,peak in enumerate(death_peaks):
      if (float(uk_df1.loc[uk_df1.index==fip,peak]) > 0):
        no_peaks_d[fip] = no_peaks_d[fip] + 1
        if fip in nearest_peak_d.keys():
            if float(uk_df1.loc[uk_df1.index==fip,peak]) > float(nearest_peak_d[fip]):
                nearest_peak_d[fip] = float(uk_df1.loc[uk_df1.index==fip,peak])
        else:
          nearest_peak_d[fip] = float(uk_df1.loc[uk_df1.index==fip,peak])
    for i,area in enumerate(death_areas):
      if (float(uk_df1.loc[uk_df1.index==fip,area]) > 0):
        if fip in nearest_area_d.keys():
            if float(uk_df1.loc[uk_df1.index==fip,area]) > float(nearest_area_d[fip]):
                nearest_area_d[fip] = float(uk_df1.loc[uk_df1.index==fip,area])
        else:                    
            nearest_area_d[fip] = float(uk_df1.loc[uk_df1.index==fip,area])
            
nearest_peak_c_df = pd.DataFrame(nearest_peak_c, index=[0]).T
nearest_area_c_df = pd.DataFrame(nearest_area_c, index=[0]).T
no_peaks_c_df = pd.DataFrame(no_peaks_c, index=[0]).T
nearest_peak_d_df = pd.DataFrame(nearest_peak_d, index=[0]).T
nearest_area_d_df = pd.DataFrame(nearest_area_d, index=[0]).T
no_peaks_d_df = pd.DataFrame(no_peaks_d, index=[0]).T

nearest_peak_c_df.rename({0:'nearest_peak_c'},axis=1,inplace=True)
nearest_area_c_df.rename({0:'nearest_area_c'},axis=1,inplace=True)
no_peaks_c_df.rename({0:'no_peaks_c'},axis=1,inplace=True)
nearest_peak_d_df.rename({0:'nearest_peak_d'},axis=1,inplace=True)
nearest_area_d_df.rename({0:'nearest_area_d'},axis=1,inplace=True)
no_peaks_d_df.rename({0:'no_peaks_d'},axis=1,inplace=True)

uk_df1 = uk_df1.merge(nearest_peak_c_df,how='left',right_index=True,left_index=True)
uk_df1 = uk_df1.merge(nearest_area_c_df,how='left',right_index=True,left_index=True)
uk_df1 = uk_df1.merge(no_peaks_c_df,how='left',right_index=True,left_index=True)
uk_df1 = uk_df1.merge(nearest_peak_d_df,how='left',right_index=True,left_index=True)
uk_df1 = uk_df1.merge(nearest_area_d_df,how='left',right_index=True,left_index=True)
uk_df1 = uk_df1.merge(no_peaks_d_df,how='left',right_index=True,left_index=True)

uk_df1.to_pickle('data/uk_df1.pkl')


## UK seat shares ##
uk_vote.set_index('ONS code',inplace=True)
uk_vote['pre-Conservative'] = uk_vote['Conservative'] - uk_vote['Conservative.2']
uk_vote['pre-Labour'] = uk_vote['Labour'] - uk_vote['Labour.2']
uk_vote['pre-LD'] = uk_vote['Liberal Democrat'] - uk_vote['Liberal Democrat.2']
uk_vote['pre-Green'] = uk_vote['Green'] - uk_vote['Green.2']
uk_vote['pre-Others'] = uk_vote['Others/independent'] - uk_vote['Others/independent.2']

uk_vote['pre-Conservative-pct'] = uk_vote['pre-Conservative']/\
    (uk_vote['pre-Conservative']+uk_vote['pre-Labour']+uk_vote['pre-LD']+uk_vote['pre-Green']+uk_vote['pre-Others'])
uk_vote['pre-Labour-pct'] = uk_vote['pre-Labour']/\
    (uk_vote['pre-Conservative']+uk_vote['pre-Labour']+uk_vote['pre-LD']+uk_vote['pre-Green']+uk_vote['pre-Others'])
uk_vote['pre-LD-pct'] = uk_vote['pre-LD']/\
    (uk_vote['pre-Conservative']+uk_vote['pre-Labour']+uk_vote['pre-LD']+uk_vote['pre-Green']+uk_vote['pre-Others'])
uk_vote['pre-Green-pct'] = uk_vote['pre-Green']/\
    (uk_vote['pre-Conservative']+uk_vote['pre-Labour']+uk_vote['pre-LD']+uk_vote['pre-Green']+uk_vote['pre-Others'])
uk_vote['pre-Others-pct'] = uk_vote['pre-Others']/\
    (uk_vote['pre-Conservative']+uk_vote['pre-Labour']+uk_vote['pre-LD']+uk_vote['pre-Green']+uk_vote['pre-Others'])
uk_vote['pre-total'] = (uk_vote['pre-Conservative']+uk_vote['pre-Labour']+uk_vote['pre-LD']+uk_vote['pre-Green']+uk_vote['pre-Others'])

uk_vote['post-total'] = (uk_vote['Conservative']+uk_vote['Labour']+uk_vote['Liberal Democrat']+\
                         uk_vote['Green']+uk_vote['Others/independent'])



# UK vote shares    
# district_onsCode = uk_vote
dist_arr = []
con_arr = {}
lab_arr = {}
other_arr = {}

# con = pd.DataFrame.from_dict(peaks_arr).T
# no_peaks_d = pd.DataFrame(no_peaks_arr, index=[0]).T

uk_vote2 = pd.read_excel("data/LEH-2021-datafile.xlsx",sheet_name='Cand_Level',header=0)

for district in uk_vote2.DISTRICTNAME.unique().tolist():
    temp = uk_vote2[uk_vote2.DISTRICTNAME==district]
    total = temp.VOTE.sum()
    other = 1.0
    if 'CON' in temp.PARTYNAME.unique().tolist():    
        num_votes = temp[temp.PARTYNAME=='CON'].VOTE.sum()
        con_arr[district] = num_votes / total
        other = other - num_votes / total
    if 'LAB' in temp.PARTYNAME.unique().tolist():    
        num_votes = temp[temp.PARTYNAME=='LAB'].VOTE.sum()
        lab_arr[district] = num_votes / total
        other = other - num_votes / total
    other_arr[district] = other

con_df = pd.DataFrame(con_arr, index=[0]).T
con_df.rename({0:'CON_2021'},axis=1,inplace=True)
lab_df = pd.DataFrame(lab_arr, index=[0]).T
lab_df.rename({0:'LAB_2021'},axis=1,inplace=True)
other_df = pd.DataFrame(con_arr, index=[0]).T
other_df.rename({0:'OTHER_2021'},axis=1,inplace=True)

temp2=uk_vote[['Local authority','ONS code']]
temp2.set_index('Local authority')
uk_df_temp = temp2.merge(con_df,how='left',right_index=True,left_on='Local authority')
uk_df_temp = uk_df_temp.merge(lab_df,how='left',right_index=True,left_on='Local authority')
uk_df_temp = uk_df_temp.merge(other_df,how='left',right_index=True,left_on='Local authority')
uk_df_temp = uk_df_temp.dropna(how='all')

nan_localities = uk_df_temp[uk_df_temp['CON_2021'].isna()]['Local authority'].to_list()
for locality in nan_localities:
    temp = uk_vote2[uk_vote2.COUNTYNAME.str.lower()==locality.lower()]
    total = temp.VOTE.sum()
    if total == 0:
        print(locality)
    else:
        other = 1.0
        num_votes = temp[temp.PARTYNAME=='CON'].VOTE.sum()
        other = other - num_votes / total
        uk_df_temp.loc[uk_df_temp['Local authority']==locality,'CON_2021'] = num_votes / total
        num_votes = temp[temp.PARTYNAME=='LAB'].VOTE.sum()
        other = other - num_votes / total
        uk_df_temp.loc[uk_df_temp['Local authority']==locality,'LAB_2021'] = num_votes / total
        num_votes = temp[temp.PARTYNAME=='OTHER'].VOTE.sum()
        uk_df_temp.loc[uk_df_temp['Local authority']==locality,'OTHER_2021'] = num_votes / total
uk_df_temp = uk_df_temp.dropna()
uk_df_temp = uk_df_temp.set_index('ONS code')

## 2017 elections
# uk_df_temp2 = pd.read_excel("data/CBP-7975-Downloadable-Results.xlsx",sheet_name='Sheet1',header=0)
uk_df_2017 = pd.read_excel("data/Local Election Summaries 2017.xlsx",sheet_name='Authority Results-NH',header=0)
uk_df_2017.rename({'Unnamed: 0':'Local authority'},axis=1,inplace=True)
uk_df_2017.rename({'Unnamed: 1':'party'},axis=1,inplace=True)
for i in range(len(uk_df_2017['Local authority'])):
    if not pd.isnull(uk_df_2017.loc[i,'Local authority']):
        temp = uk_df_2017.loc[i,'Local authority']
    else:
        uk_df_2017.loc[i,'Local authority'] = temp
len(set(uk_df_temp['Local authority']).intersection(uk_df_2017['Local authority'].unique()))
set(uk_df_temp['Local authority']).intersection(uk_df_2017['Local authority'].unique())


## 2016 elections
uk_df_2016 = pd.read_excel("data/Local Election Summaries 2016.xlsx",sheet_name='Authority Results-NH',header=0)
uk_df_2016.rename({'Unnamed: 0':'Local authority'},axis=1,inplace=True)
uk_df_2016.rename({'Unnamed: 1':'party'},axis=1,inplace=True)
for i in range(len(uk_df_2016['Local authority'])):
    if not pd.isnull(uk_df_2016.loc[i,'Local authority']):
        temp = uk_df_2016.loc[i,'Local authority']
    else:
        uk_df_2016.loc[i,'Local authority'] = temp
len(set(uk_df_temp['Local authority']).intersection(uk_df_2016['Local authority'].unique()))
len(set(uk_df_2017['Local authority']).intersection(uk_df_2016['Local authority'].unique()))
set(uk_df_temp['Local authority']).intersection(uk_df_2016['Local authority'].unique())
len(set(uk_df_temp['Local authority']).intersection(uk_2016_2017))
len(set(uk_df_temp.index).intersection(con_df.index))
len(set(uk_df_temp.index).difference(con_df.index))
len(set(uk_df_temp['Local authority']).intersection(uk_2016_2017))
set(uk_df_temp['Local authority']).difference(uk_2016_2017)

locality_ONS = {uk_df_temp['Local authority'][i]: uk_df_temp.index[i] for i in range(len(uk_df_temp['Local authority']))}
locality_ONS['Isle Of Wight']=locality_ONS['Isle of Wight'] #to add another spelling of Isle of Wight 
ONS_locality = {uk_df_temp.index[i]: uk_df_temp['Local authority'][i] for i in range(len(uk_df_temp.index))}

## combine 2016,2017 results to 2021 results
con_arr = {}
lab_arr = {}
other_arr = {}
year = {}

for local in uk_df_2016['Local authority'].unique().tolist():
    if local in locality_ONS.keys():        
        temp = uk_df_2016[uk_df_2016['Local authority']==local]
        other = 1.0
        if 'Con' in temp.party.unique().tolist():        
            con_arr[locality_ONS[local]] = temp.loc[temp.party=='Con','% Share'].to_list()[0]/100
            other = other - con_arr[locality_ONS[local]]
        if 'Lab' in temp.party.unique().tolist():        
            lab_arr[locality_ONS[local]] = temp.loc[temp.party=='Lab','% Share'].to_list()[0]/100
            other = other - lab_arr[locality_ONS[local]]
        year[locality_ONS[local]] = 2016
        other_arr[locality_ONS[local]] = other

for local in uk_df_2017['Local authority'].unique().tolist():
    if local in locality_ONS.keys():        
        temp = uk_df_2017[uk_df_2017['Local authority']==local]
        other = 1.0
        if 'Con' in temp.party.unique().tolist():        
            con_arr[locality_ONS[local]] = temp.loc[temp.party=='Con','% Share'].to_list()[0]/100
            other = other - con_arr[locality_ONS[local]]
        if 'Lab' in temp.party.unique().tolist():        
            lab_arr[locality_ONS[local]] = temp.loc[temp.party=='Lab','% Share'].to_list()[0]/100
            other = other - lab_arr[locality_ONS[local]]
        year[locality_ONS[local]] = 2017
        other_arr[locality_ONS[local]] = other

con_df = pd.DataFrame(con_arr, index=[0]).T
con_df.rename({0:'CON_2016_17'},axis=1,inplace=True)
lab_df = pd.DataFrame(lab_arr, index=[0]).T
lab_df.rename({0:'LAB_2016_17'},axis=1,inplace=True)
other_df = pd.DataFrame(con_arr, index=[0]).T
other_df.rename({0:'OTHER_2016_17'},axis=1,inplace=True)
year_df = pd.DataFrame(year, index=[0]).T
year_df.rename({0:'year_last_election'},axis=1,inplace=True)

uk_df_temp2 = uk_df_temp.merge(con_df,how='left',right_index=True,left_index=True)
uk_df_temp2 = uk_df_temp2.merge(lab_df,how='left',right_index=True,left_index=True)
uk_df_temp2 = uk_df_temp2.merge(other_df,how='left',right_index=True,left_index=True)
uk_df_temp2 = uk_df_temp2.merge(year_df,how='left',right_index=True,left_index=True)
uk_df_temp2 = uk_df_temp2.dropna(how='all')
# uk_df_temp2.to_pickle('data/uk_df_temp2.pkl')

# only 111 out of 114 had elections in 2016 or 2017
len(set(uk_df_temp2.index).intersection(uk_df4.index))
set(uk_df4.index).difference(uk_df_temp2.index)

uk_df5 = uk_df4.merge(uk_df_temp2,how='inner',right_index=True,left_index=True)
# uk_df5.to_pickle('data/uk_df5.pkl')










uk_df2 = uk_df1.merge(uk_vote,how='inner',right_index=True,left_index=True)

uk_df2['post-total'] = (uk_df2['Conservative']+uk_df2['Labour']+uk_df2['Liberal Democrat']+\
                         uk_df2['Green']+uk_df2['Others/independent'])
uk_df2['Conservative-pct'] = uk_df2['Conservative']/uk_df2['post-total']
uk_df2['Labour-pct'] = uk_df2['Labour']/uk_df2['post-total']
uk_df2['LD-pct'] = uk_df2['Liberal Democrat']/uk_df2['post-total']
uk_df2['Green-pct'] = uk_df2['Green']/uk_df2['post-total']
uk_df2['Others-pct'] = uk_df2['Others/independent']/uk_df2['post-total']

uk_df2['Conservative-diff-pct'] = uk_df2['Conservative-pct'] - uk_df2['pre-Conservative-pct']
uk_df2['Labour-diff-pct'] = uk_df2['Labour-pct'] - uk_df2['pre-Labour-pct']
uk_df2['LD-votes-pct'] = uk_df2['LD-pct'] - uk_df2['pre-LD-pct']
uk_df2['Green-diff-pct'] = uk_df2['Green-pct']-uk_df2['pre-Green-pct']
uk_df2['Others-diff-pct'] = uk_df2['Others-pct'] - uk_df2['pre-Others-pct']

uk_df2['Conservative_votes_diff_diff'] = uk_df2['Conservative-diff-pct'] - uk_df2['Labour-diff-pct'] -\
    uk_df2['LD-votes-pct'] - uk_df2['Green-diff-pct'] - uk_df2['Others-diff-pct']
uk_df2['Labour_votes_diff_diff'] = uk_df2['Labour-diff-pct'] -uk_df2['Conservative-diff-pct']-\
    uk_df2['LD-votes-pct'] - uk_df2['Green-diff-pct'] - uk_df2['Others-diff-pct']
uk_df2['LD_votes_diff_diff'] = uk_df2['LD-votes-pct'] - uk_df2['Labour-diff-pct'] -\
    uk_df2['Conservative-diff-pct']-uk_df2['Green-diff-pct'] - uk_df2['Others-diff-pct']
uk_df2['Green_votes_diff_diff'] = uk_df2['Green-diff-pct'] - uk_df2['LD-votes-pct'] - \
    uk_df2['Labour-diff-pct'] -uk_df2['Conservative-diff-pct']-uk_df2['Others-diff-pct']

pre_winning_party = {}
cols = ['pre-Conservative-pct','pre-Labour-pct','pre-LD-pct','pre-Green-pct','pre-Others-pct']
parties = ['Conservative','Labour','LD','Green','Others']
for fip in uk_df2.index:
    temp = uk_df2.loc[uk_df2.index==fip,cols]
    pre_winning_party[fip] = parties[np.argmax(temp)]        

pre_winning_party_df = pd.DataFrame(pre_winning_party, index=[0]).T
pre_winning_party_df.rename({0:'pre_winning_party'},axis=1,inplace=True)

uk_df2 = uk_df2.merge(pre_winning_party_df,how='left',right_index=True,left_index=True)
# uk_df2.to_pickle('data/uk_df2.pkl')
# uk_df4.to_pickle('data/uk_df4.pkl')
# uk_df4.to_pickle('data/uk_df4.pkl')
# uk_df2= pd.read_pickle("data/uk_df2.pkl")
uk_df4 = pd.read_pickle("data/uk_df4.pkl")



## Conservative  Strong support case area 
temp = uk_df2[uk_df2.pre_winning_party == 'Conservative']
Y, X1 = dmatrices('Conservative-votes_diff-diff ~ nearest_peak_c + nearest_peak_d + nearest_area_d + \
                              nearest_area_c + no_peaks_c + no_peaks_d'\
                                      ,temp, return_type="dataframe")
Y = np.ravel(Y)

## Labour  mixed: support from case area, opposite from case peak and death area
temp = uk_df2[uk_df2.pre_winning_party == 'Labour']
Y, X1 = dmatrices('Labour-votes_diff-diff ~ nearest_peak_c + nearest_peak_d + nearest_area_d + \
                              nearest_area_c + no_peaks_c + no_peaks_d'\
                                      ,temp, return_type="dataframe")
Y = np.ravel(Y)

## Liberal Democrat mixed: support from case area, opposite from case peak and death area
temp = uk_df2[uk_df2.pre_winning_party == 'LD']
Y, X1 = dmatrices('LD_votes_diff_diff ~ nearest_peak_c + nearest_peak_d + nearest_area_d + \
                              nearest_area_c + no_peaks_c + no_peaks_d'\
                                      ,temp, return_type="dataframe")
Y = np.ravel(Y)

## Green mixed: support from case area, opposite from case peak and death area
temp = uk_df3[uk_df3.pre_winning_party == 'Green']
Y, X1 = dmatrices('Green-votes_diff-diff ~ nearest_peak_c + nearest_peak_d + nearest_area_d + \
                              nearest_area_c + no_peaks_c + no_peaks_d'\
                                      ,temp, return_type="dataframe")
Y = np.ravel(Y)

model1 = sm.OLS(Y,X1.astype(float))
results1 = model1.fit()
coefs1 = pd.DataFrame({
    'coef': results1.params.values,
    'stderr': results1.bse,
    'pvalue': results1.pvalues
}).sort_values(by='pvalue', ascending=True)
coefs1.loc['nearest_area_d',:]
coefs1.loc['nearest_area_c',:]
coefs1.loc['nearest_peak_d',:]
coefs1.loc['nearest_peak_c',:]
coefs1.loc['no_peaks_c',:]

uk_df4.to_csv('data/uk_data.csv')



############################################################################
################################ JOP submission ############################
################################ UK covariates ############################
############################################################################

uk_df4 = pd.read_pickle("data/uk_df4.pkl")


## Conservative  Strong support case area 
# Y, X1 = dmatrices('Conservative_votes_diff_diff ~ nearest_peak_c + nearest_peak_d + nearest_area_d + \
#                               nearest_area_c + no_peaks_c + no_peaks_d + smokers + weekly_pay +\
#                                   employment_rate+overweight_adults + pop_65_up_pct +\
#                                       pop_20_29_pct + pop_30_39_pct',
#                               temp, return_type="dataframe")

    
temp4 = uk_df4[uk_df4.pre_winning_party == 'Conservative']

Y4, X4 = dmatrices('Conservative-diff-pct ~ nearest_peak_c +nearest_peak_d + smokers +\
                   weekly_pay + employment_rate+ overweight_adults + pop_65_up_pct +\
                       pop_20_29_pct + pop_30_39_pct', temp4, return_type="dataframe")
    
Y4 = np.ravel(Y4)
model4 = sm.OLS(Y4,X4.astype(float))
results4 = model4.fit()
coefs4 = pd.DataFrame({
    'coef': results4.params.values,
    'stderr': results4.bse,
    'pvalue': results4.pvalues
}).sort_values(by='pvalue', ascending=True)

coefs4.loc['nearest_peak_c',:]
coefs4.loc['nearest_peak_d',:]

# Y, X1 = dmatrices('Conservative_votes_diff_diff ~ peak1_c + peak1_d+ area1_d + \
#                               area1_c + no_peaks_c + no_peaks_d + smokers + weekly_pay+\
#                                   employment_rate+overweight_adults',
#                               temp, return_type="dataframe")



# Y, X1 = dmatrices('Labour_votes_diff_diff ~ nearest_peak_c + nearest_peak_d + nearest_area_d + \
#                               nearest_area_c + no_peaks_c + no_peaks_d + smokers + weekly_pay +\
#                                   employment_rate + overweight_adults + pop_65_up_pct +\
#                                       pop_20_29_pct + pop_30_39_pct',
#                               temp, return_type="dataframe")
temp3 = uk_df4[uk_df4.pre_winning_party == 'Labour']
Y3, X3 = dmatrices('Labour-diff-pct ~ nearest_peak_c + nearest_peak_d+\
                              smokers + weekly_pay +\
                                  employment_rate+overweight_adults + pop_65_up_pct +\
                                      pop_20_29_pct + pop_30_39_pct',
                              temp3, return_type="dataframe")
    
# Y, X1 = dmatrices('Labour_votes_diff_diff ~ peak1_c + peak1_d+ area1_d + \
#                               area1_c + no_peaks_c + no_peaks_d + smokers + weekly_pay +\
#                                   employment_rate + overweight_adults',
#                               temp, return_type="dataframe")

Y3 = np.ravel(Y3)
model3 = sm.OLS(Y3,X3.astype(float))
results3 = model3.fit()
coefs3 = pd.DataFrame({
    'coef': results3.params.values,
    'stderr': results3.bse,
    'pvalue': results3.pvalues
}).sort_values(by='pvalue', ascending=True)

coefs1.loc['nearest_peak_c',:]
coefs1.loc['nearest_peak_d',:]

# coefs1.loc['area1_d',:]
# coefs1.loc['area1_c',:]
# coefs1.loc['peak1_d',:]
# coefs1.loc['peak1_c',:]
# coefs1.loc['no_peaks_c',:]
# coefs1.loc['nearest_area_d',:]
# coefs1.loc['nearest_area_c',:]
# coefs1.loc['no_peaks_c',:]
# coefs1.loc['pre_winning_party',:]



temp = uk_df3[['pre_winning_party',"Conservative_votes_diff_diff","Labour_votes_diff_diff",
               "nearest_peak_c","nearest_peak_d","nearest_area_d",
               "nearest_area_c","no_peaks_c","no_peaks_d","smokers",'weekly_pay']]
temp = uk_df3[["Conservative_votes_diff_diff","no_peaks_d"]]
sns.pairplot(temp)
temp.to_csv('data/data.csv')

uk_df4 = uk_df3[(uk_df3.Conservative_votes_diff_diff!=0) & (uk_df3.Labour_votes_diff_diff!=0)]


## UK graph analysis
temp = uk_df4[uk_df4.pre_winning_party == 'Conservative']
conservative = temp.loc[temp.Conservative_votes_diff_diff>0.1,['Local authority','Class of authority','Conservative_votes_diff_diff','tot_pop','pre-Conservative-pct','Conservative-pct','Conservative-diff-pct','pre-Labour-pct','Labour-pct','Labour-diff-pct','pre-LD-pct','LD-pct','LD-votes-pct','pre-Green-pct','Green-pct','Green-diff-pct','pre-Others-pct','Others-pct','Others-diff-pct', 'no_peaks_c','no_peaks_d','nearest_peak_c','nearest_peak_d','nearest_area_c','nearest_area_d']]

plot_kr_scatter_uk(['E07000236','E07000229'],type='cases')