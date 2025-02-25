# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 20:57:55 2022

@author: neilh
"""


"""# Data Construction

## States
"""

state_demo = pd.read_csv("data/DECENNIALPL2020.P1_data_with_overlays_2022-01-19T210012.csv")
# state_region_xwalk = county_df6[['state_po','CensusRegionName','CensusDivisionName']].drop_duplicates()
# state_region_xwalk.set_index('state_po',drop=True,inplace=True)
# state_region_xwalk.to_pickle('data/state_region_xwalk.pkl')

covidSt = covidSt[covid.date<'2020-11-03']

# for state_po in list(covidSt.state_po.unique()):
#     covidSt_plot = covidSt[covidSt.state_po== state_po]
#     covidSt_plot["daily_d"] = covidSt_plot.deaths_n.diff()
#     covidSt_plot["daily_c"] = covidSt_plot.cases_n.diff()
#     covidSt_plot.loc[covidSt_plot.daily_d<0,'daily_d'] = 0
#     covidSt_plot.loc[covidSt_plot.daily_c<0,'daily_c'] = 0
#     covidSt_plot["ma_d"] = np.rint(covidSt_plot.daily_d.rolling(window=wl).mean())
#     covidSt_plot["ma_c"] = np.rint(covidSt_plot.daily_c.rolling(window=wl).mean())
#     covidSt_plot = covidSt_plot.iloc[wl:]
#     if (covidSt_plot.shape[0]>0):
#       max_of_d_and_c = np.max([covidSt_plot.ma_d,covidSt_plot.ma_c])
#       if max_of_d_and_c*1.1 > max_yaxis:
#           max_yaxis = max_of_d_and_c*1.1
#       if covidSt_plot.index.shape[0]+wl > max_xaxis:
#           max_xaxis = covidSt_plot.index.shape[0]+wl

# print(max_xaxis, max_yaxis)

max_xaxis = 210 
max_yaxis = 6079.700000000001
wl = 7

peaks_arr = {}
widths_arr = {}
heights_arr = {}
MBdim_arr = {}
MBdimKR_arr = {}
areas_arr = {}
areas_KR_arr = {}
no_peaks_arr = {}

beta = 0.3
max_no_height =8

for state_po in list(covidSt.state_po.unique()):
  covidSt_plot = covidSt[covidSt.state_po== state_po]    
  zero_df = pd.DataFrame(0, index=np.arange(pad_zeros), columns=covidSt_plot.columns)
  pad_zeros = max_xaxis - covidSt_plot.shape[0]
  covidSt_plot = pd.concat([zero_df,covidSt_plot])
  # directory = 'plots_states/cases/'; covidSt_plot["daily"] = covidSt_plot.cases_n.diff()
  directory = 'plots_states/deaths/'; covidSt_plot["daily"] = covidSt_plot.deaths_n.diff()
  covidSt_plot.loc[covidSt_plot.daily<0,'daily'] = 0 
  covidSt_plot["ma"] = np.rint(covidSt_plot.daily.rolling(window=wl).mean())
  covidSt_plot = covidSt_plot.iloc[wl:,:]
  covidSt_plot = covidSt_plot.reset_index()
  covidSt_plot['periods'] = covidSt_plot.index        
  xTrain = covidSt_plot.periods.to_numpy()[:,np.newaxis]
  yTrain = covidSt_plot.ma
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
  peaks_arr[state_po] = np.pad(peaks[idx],pad_zeros,mode='constant')[pad_zeros:]
  widths_arr[state_po] = np.pad(widths,pad_zeros,mode='constant')[pad_zeros:]
  heights_arr[state_po] = np.pad(heights,pad_zeros,mode='constant')[pad_zeros:]
  no_peaks_arr[state_po] = no_idx
  areas = []
  areas_kr = []
  for ind in idx:
    start_ind = max(0,peaks[ind]-int(np.round(w[ind]/2.0)))
    end_ind = min(len(yHatk),peaks[ind]+int(np.round(w[ind]/2.0)+1))
    yvalues_kr = yHatk[start_ind:end_ind]
    if min(yvalues_kr)<0:
      yvalues_kr = yvalues_kr - min(yvalues_kr)
    yvalues = covidSt_plot.daily[start_ind:end_ind]
    if min(yvalues)<0:
      yvalues = yvalues - min(yvalues)
    areas.append(sum(yvalues))
    areas_kr.append(sum(yvalues_kr))      
  areas_arr[state_po] = np.pad(areas,pad_zeros,mode='constant')[pad_zeros:]
  areas_KR_arr[state_po] = np.pad(areas_kr,pad_zeros,mode='constant')[pad_zeros:]
  
  fig, ax = plt.subplots( nrows=1, ncols=1, figsize=(30,16) )
  ax.plot(xTrain,yTrain)
  ax.axis('off')
  fig.savefig(directory+state_po+'.png')
  plt.close(fig)
  fig.clear()
  I = (imageio.imread(directory+state_po+'.png')/256.0)[:,:,0] #take gray scale
  MBdim_arr[state_po] = fractal_dimension(I)
  fig, ax = plt.subplots( nrows=1, ncols=1, figsize=(30,16) )
  ax.plot(xTrain,yHatk)
  ax.axis('off')
  fig.savefig(directory+state_po+'_KR.png')
  fig.clear()
  plt.close(fig)
  I = (imageio.imread(directory+state_po+'_KR.png')/256.0)[:,:,0] #take gray scale
  MBdimKR_arr[state_po] = fractal_dimension(I)
  print(str(len(heights_arr)) +' ' + state_po)
  if len(heights_arr) % 250 == 0:
    gc.collect()

# peaks_c = pd.DataFrame.from_dict(peaks_arr).T
# widths_c = pd.DataFrame.from_dict(widths_arr).T
# heights_c = pd.DataFrame.from_dict(heights_arr).T
# areas_c = pd.DataFrame.from_dict(areas_arr).T
# areas_KR_c = pd.DataFrame.from_dict(areas_KR_arr).T
# no_peaks_c = pd.DataFrame(no_peaks_arr, index=[0]).T
# MBdim_c = pd.DataFrame(MBdim_arr, index=[0]).T
# MBdimKR_c = pd.DataFrame(MBdimKR_arr, index=[0]).T

# peaks_c.to_pickle("data/states/peaks_cases.pkl")
# widths_c.to_pickle("data/states/widths_cases.pkl")
# heights_c.to_pickle("data/states/heights_cases.pkl")
# areas_c.to_pickle("data/states/areas_cases.pkl")
# areas_KR_c.to_pickle("data/states/areas_KR_cases.pkl")
# MBdim_c.to_pickle("data/states/MBdim_cases.pkl")
# MBdimKR_c.to_pickle("data/states/MBdimKR_cases.pkl")
# no_peaks_c.to_pickle("data/states/no_peaks_cases.pkl")

# peaks_d = pd.DataFrame.from_dict(peaks_arr).T
# widths_d = pd.DataFrame.from_dict(widths_arr).T
# heights_d = pd.DataFrame.from_dict(heights_arr).T
# areas_d = pd.DataFrame.from_dict(areas_arr).T
# areas_KR_d = pd.DataFrame.from_dict(areas_KR_arr).T
# no_peaks_d = pd.DataFrame(no_peaks_arr, index=[0]).T
# MBdim_d = pd.DataFrame(MBdim_arr, index=[0]).T
# MBdimKR_d = pd.DataFrame(MBdimKR_arr, index=[0]).T

# peaks_d.to_pickle("data/states/peaks_deaths.pkl")
# widths_d.to_pickle("data/states/widths_deaths.pkl")
# heights_d.to_pickle("data/states/heights_deaths.pkl")
# areas_d.to_pickle("data/states/areas_deaths.pkl")
# areas_KR_d.to_pickle("data/states/areas_KR_deaths.pkl")
# no_peaks_d.to_pickle("data/states/no_peaks_deaths.pkl")
# MBdim_d.to_pickle("data/states/MBdim_deaths.pkl")
# MBdimKR_d.to_pickle("data/states/MBdimKR_deaths.pkl")

# # Death counts
# peaks_d.rename({0:'peak1_d',1:'peak2_d',2:'peak3_d',3:'peak4_d',4:'peak5_d',5:'peak6_d',6:'peak7_d',7:'peak8_d'}, axis=1, inplace=True)
# widths_d.rename({0:'width1_d',1:'width2_d',2:'width3_d',3:'width4_d',4:'width5_d',5:'width6_d',6:'width7_d',7:'width8_d'}, axis=1, inplace=True)
# heights_d.rename({0:'height1_d',1:'height2_d',2:'height3_d',3:'height4_d',4:'height5_d',5:'height6_d',6:'height7_d',7:'height8_d'}, axis=1, inplace=True)
# areas_d.rename({0:'area1_d',1:'area2_d',2:'area3_d',3:'area4_d',4:'area5_d',5:'area6_d',6:'area7_d',7:'area8_d'}, axis=1, inplace=True)
# areas_KR_d.rename({0:'area1_KR_d',1:'area2_KR_d',2:'area3_KR_d',3:'area4_KR_d',4:'area5_KR_d',5:'area6_KR_d',6:'area7_KR_d',7:'area8_KR_d'}, axis=1, inplace=True)
# MBdim_d.rename({0:'MBdim_d'}, axis=1, inplace=True)
# MBdimKR_d.rename({0:'MBdimKR_d'}, axis=1, inplace=True)
# no_peaks_c.rename({0:'no_peaks_c'}, axis=1, inplace=True)

# peaks_d.to_pickle("data/states/peaks_deaths.pkl")
# widths_d.to_pickle("data/states/widths_deaths.pkl")
# heights_d.to_pickle("data/states/heights_deaths.pkl")
# areas_d.to_pickle("data/states/areas_deaths.pkl")
# areas_KR_d.to_pickle("data/states/areas_KR_deaths.pkl")
# MBdim_d.to_pickle("data/states/MBdim_d.pkl")
# MBdimKR_d.to_pickle("data/states/MBdimKR_d.pkl")
# no_peaks_c.to_pickle("data/states/no_peaks_c.pkl")

# # Cases
# peaks_c.rename({0:'peak1_c',1:'peak2_c',2:'peak3_c',3:'peak4_c',4:'peak5_c',5:'peak6_c',6:'peak7_c',7:'peak8_c'}, axis=1, inplace=True)
# widths_c.rename({0:'width1_c',1:'width2_c',2:'width3_c',3:'width4_c',4:'width5_c',5:'width6_c',6:'width7_c',7:'width8_c'}, axis=1, inplace=True)
# heights_c.rename({0:'height1_c',1:'height2_c',2:'height3_c',3:'height4_c',4:'height5_c',5:'height6_c',6:'height7_c',7:'height8_c'}, axis=1, inplace=True)
# areas_c.rename({0:'area1_c',1:'area2_c',2:'area3_c',3:'area4_c',4:'area5_c',5:'area6_c',6:'area7_c',7:'area8_c'}, axis=1, inplace=True)
# areas_KR_c.rename({0:'area1_KR_c',1:'area2_KR_c',2:'area3_KR_c',3:'area4_KR_c',4:'area5_KR_c',5:'area6_KR_c',6:'area7_KR_c',7:'area8_KR_c'}, axis=1, inplace=True)
# MBdim_c.rename({0:'MBdim_c'}, axis=1, inplace=True)
# MBdimKR_c.rename({0:'MBdimKR_c'}, axis=1, inplace=True)
# no_peaks_d.rename({0:'no_peaks_d'}, axis=1, inplace=True)

# peaks_c.to_pickle("data/states/peaks_cases.pkl")
# widths_c.to_pickle("data/states/widths_cases.pkl")
# heights_c.to_pickle("data/states/heights_cases.pkl")
# areas_c.to_pickle("data/states/areas_cases.pkl")
# areas_KR_c.to_pickle("data/states/areas_KR_cases.pkl")
# MBdim_c.to_pickle("data/states/MBdim_c.pkl")
# MBdimKR_c.to_pickle("data/states/MBdimKR_c.pkl")
# no_peaks_d.to_pickle("data/states/no_peaks_d.pkl")

# peaks_c = pd.read_pickle("data/states/peaks_cases.pkl")
# widths_c = pd.read_pickle("data/states/widths_cases.pkl")
# heights_c = pd.read_pickle("data/states/heights_cases.pkl")
# MBdim_c = pd.read_pickle("data/states/MBdim_cases.pkl")
# MBdimKR_c = pd.read_pickle("data/states/MBdimKR_cases.pkl")
# no_peaks_c = pd.read_pickle("data/states/no_peaks_cases.pkl")

# peaks_d = pd.read_pickle("data/states/peaks_deaths.pkl")
# widths_d = pd.read_pickle("data/states/widths_deaths.pkl")
# heights_d = pd.read_pickle("data/states/heights_deaths.pkl")
# MBdim_d = pd.read_pickle("data/states/MBdim_deaths.pkl")
# MBdimKR_d = pd.read_pickle("data/states/MBdimKR_deaths.pkl")
# no_peaks_d = pd.read_pickle("data/states/no_peaks_deaths.pkl")

covidSt_df1 = peaks_c.merge(widths_c,how='inner',right_index=True,left_index=True)
covidSt_df1 = covidSt_df1.merge(heights_c,how='inner',right_index=True,left_index=True)
covidSt_df1 = covidSt_df1.merge(areas_c,how='inner',right_index=True,left_index=True)
covidSt_df1 = covidSt_df1.merge(areas_KR_c,how='inner',right_index=True,left_index=True)
covidSt_df1 = covidSt_df1.merge(MBdim_c,how='inner',right_index=True,left_index=True)
covidSt_df1 = covidSt_df1.merge(MBdimKR_c,how='inner',right_index=True,left_index=True)
covidSt_df1 = covidSt_df1.merge(no_peaks_c,how='inner',right_index=True,left_index=True)

covidSt_df1 = covidSt_df1.merge(peaks_d,how='inner',right_index=True,left_index=True)
covidSt_df1 = covidSt_df1.merge(widths_d,how='inner',right_index=True,left_index=True)
covidSt_df1 = covidSt_df1.merge(heights_d,how='inner',right_index=True,left_index=True)
covidSt_df1 = covidSt_df1.merge(areas_d,how='inner',right_index=True,left_index=True)
covidSt_df1 = covidSt_df1.merge(areas_KR_d,how='inner',right_index=True,left_index=True)
covidSt_df1 = covidSt_df1.merge(MBdim_d,how='inner',right_index=True,left_index=True)
covidSt_df1 = covidSt_df1.merge(MBdimKR_d,how='inner',right_index=True,left_index=True)
covidSt_df1 = covidSt_df1.merge(no_peaks_d,how='inner',right_index=True,left_index=True)

covidSt_df1.iloc[43:50,:]
# covidSt_df1.to_pickle("data/covidSt_df1.pkl")

temp1=np.zeros(covidSt_df1.shape[0])
temp2=np.zeros(covidSt_df1.shape[0])
temp3=np.zeros(covidSt_df1.shape[0])
temp4=np.zeros(covidSt_df1.shape[0])
temp5=np.zeros(covidSt_df1.shape[0])
temp6=np.zeros(covidSt_df1.shape[0])
temp7=np.zeros(covidSt_df1.shape[0])
temp8=np.zeros(covidSt_df1.shape[0])
temp = [0,0,0,0,0,0,0,0]

for i in range(covidSt_df1.shape[0]):  
    x = covidSt_df1[['peak1_c','peak2_c','peak3_c','peak4_c','peak5_c','peak6_c','peak7_c','peak8_c']][covidSt_df1[['peak1_c','peak2_c','peak3_c','peak4_c','peak5_c','peak6_c','peak7_c','peak8_c']] > 0].iloc[i,:].nsmallest(8)
    temp[:len(x)] = x
    temp1[i] = temp[0]
    temp2[i] = temp[1]
    temp3[i] = temp[2]
    temp4[i] = temp[3]
    temp5[i] = temp[4]
    temp6[i] = temp[5]
    temp7[i] = temp[6]
    temp8[i] = temp[7]
    temp = [0,0,0,0,0,0,0,0]
    
covidSt_df1.loc[:,'localmax1_c'] = temp1
covidSt_df1.loc[:,'localmax2_c'] = temp2
covidSt_df1.loc[:,'localmax3_c'] = temp3
covidSt_df1.loc[:,'localmax4_c'] = temp4
covidSt_df1.loc[:,'localmax5_c'] = temp5
covidSt_df1.loc[:,'localmax6_c'] = temp6
covidSt_df1.loc[:,'localmax7_c'] = temp7
covidSt_df1.loc[:,'localmax8_c'] = temp8

for i in range(covidSt_df1.shape[0]):
    x = covidSt_df1[['peak1_d','peak2_d','peak3_d','peak4_d','peak5_d','peak6_d','peak7_d','peak8_d']][covidSt_df1[['peak1_d','peak2_d','peak3_d','peak4_d','peak5_d','peak6_d','peak7_d','peak8_d']] > 0].iloc[i,:].nsmallest(8)
    temp[:len(x)] = x
    temp1[i] = temp[0]
    temp2[i] = temp[1]
    temp3[i] = temp[2]
    temp4[i] = temp[3]
    temp5[i] = temp[4]
    temp6[i] = temp[5]
    temp7[i] = temp[6]
    temp8[i] = temp[7]
    temp = [0,0,0,0,0,0,0,0]

covidSt_df1.loc[:,'localmax1_d'] = temp1
covidSt_df1.loc[:,'localmax2_d'] = temp2
covidSt_df1.loc[:,'localmax3_d'] = temp3
covidSt_df1.loc[:,'localmax4_d'] = temp4
covidSt_df1.loc[:,'localmax5_d'] = temp5
covidSt_df1.loc[:,'localmax6_d'] = temp6
covidSt_df1.loc[:,'localmax7_d'] = temp7
covidSt_df1.loc[:,'localmax8_d'] = temp8

vars = ['FracMale2017','MedianAge2010','HeartDiseaseMortality','PopulationEstimate2018',\
        'Smokers_Percentage','RespMortalityRate2014','DiabetesPercentage','unemployed_pct',\
        'single_parent_HH_pct','pop65plus','Obesity_pct']

FracMale2017_arr = {}
MedianAge2010_arr = {}
HeartDiseaseMortality_arr = {}
PopulationEstimate2018_arr = {}
Smokers_Percentage_arr = {}
RespMortalityRate2014_arr = {}
DiabetesPercentage_arr = {}
unemployed_pct_arr = {}
single_parent_HH_pct_arr = {}
pop65plus_arr = {}
Obesity_pct_arr = {}
rural_urban_arr = {}

for state_po in list(county_df6.state_po.unique()):
# for state_po in ['AR','AL']:
  temp = county_df6[county_df6.state_po== state_po]
  tot_pop = np.sum(temp['PopulationEstimate2018'])
  temp['weight'] = temp['PopulationEstimate2018']/tot_pop
  FracMale2017_arr[state_po] = temp['FracMale2017'].dot(temp['weight'])
  MedianAge2010_arr[state_po] = temp['MedianAge2010'].dot(temp['weight'])
  HeartDiseaseMortality_arr[state_po] = temp['HeartDiseaseMortality'].dot(temp['weight'])
  PopulationEstimate2018_arr[state_po] = np.sum(temp['PopulationEstimate2018'])
  Smokers_Percentage_arr[state_po] = temp['Smokers_Percentage'].dot(temp['weight'])
  RespMortalityRate2014_arr[state_po] = temp['RespMortalityRate2014'].dot(temp['weight'])
  DiabetesPercentage_arr[state_po] = temp['DiabetesPercentage'].dot(temp['weight'])
  unemployed_pct_arr[state_po] = temp['unemployed_pct'].dot(temp['weight'])
  single_parent_HH_pct_arr[state_po] = temp['single_parent_HH_pct'].dot(temp['weight'])
  pop65plus_arr[state_po] = temp['pop65plus'].dot(temp['weight'])
  Obesity_pct_arr[state_po] = temp['Obesity_pct'].dot(temp['weight'])
  rural_urban_arr[state_po] = temp['Rural_UrbanContinuumCode2013'].dot(temp['weight'])

FracMale2017_df = pd.DataFrame(FracMale2017_arr, index=[0]).T
MedianAge2010_df = pd.DataFrame(MedianAge2010_arr, index=[0]).T
HeartDiseaseMortality_df = pd.DataFrame(HeartDiseaseMortality_arr, index=[0]).T
PopulationEstimate2018_df = pd.DataFrame(PopulationEstimate2018_arr, index=[0]).T
Smokers_Percentage_df = pd.DataFrame(Smokers_Percentage_arr, index=[0]).T
RespMortalityRate2014_df = pd.DataFrame(RespMortalityRate2014_arr, index=[0]).T
DiabetesPercentage_df = pd.DataFrame(DiabetesPercentage_arr, index=[0]).T
unemployed_pct_df = pd.DataFrame(unemployed_pct_arr, index=[0]).T
single_parent_HH_pct_df = pd.DataFrame(single_parent_HH_pct_arr, index=[0]).T
pop65plus_df = pd.DataFrame(pop65plus_arr, index=[0]).T
Obesity_pct_df = pd.DataFrame(Obesity_pct_arr, index=[0]).T
rural_urban_df = pd.DataFrame(rural_urban_arr, index=[0]).T

FracMale2017_df.rename({0:'FracMale2017'},axis=1,inplace=True)
MedianAge2010_df.rename({0:'MedianAge2010'},axis=1,inplace=True)
HeartDiseaseMortality_df.rename({0:'HeartDiseaseMortality'},axis=1,inplace=True)
PopulationEstimate2018_df.rename({0:'PopulationEstimate2018'},axis=1,inplace=True)
Smokers_Percentage_df.rename({0:'Smokers_Percentage'},axis=1,inplace=True)
RespMortalityRate2014_df.rename({0:'RespMortalityRate2014'},axis=1,inplace=True)
DiabetesPercentage_df.rename({0:'DiabetesPercentage'},axis=1,inplace=True)
unemployed_pct_df.rename({0:'unemployed_pct'},axis=1,inplace=True)
single_parent_HH_pct_df.rename({0:'single_parent_HH_pct'},axis=1,inplace=True)
pop65plus_df.rename({0:'pop65plus'},axis=1,inplace=True)
Obesity_pct_df.rename({0:'Obesity_pct'},axis=1,inplace=True)
rural_urban_df.rename({0:'rural_urban'},axis=1,inplace=True)

emotional_health2 = emotional_health.merge(FracMale2017_df,how='left',left_on='state_po',right_index=True)
emotional_health2 = emotional_health2.merge(MedianAge2010_df,how='left',left_on='state_po',right_index=True)
emotional_health2 = emotional_health2.merge(HeartDiseaseMortality_df,how='left',left_on='state_po',right_index=True)
emotional_health2 = emotional_health2.merge(PopulationEstimate2018_df,how='left',left_on='state_po',right_index=True)
emotional_health2 = emotional_health2.merge(Smokers_Percentage_df,how='left',left_on='state_po',right_index=True)
emotional_health2 = emotional_health2.merge(RespMortalityRate2014_df,how='left',left_on='state_po',right_index=True)
emotional_health2 = emotional_health2.merge(DiabetesPercentage_df,how='left',left_on='state_po',right_index=True)
emotional_health2 = emotional_health2.merge(unemployed_pct_df,how='left',left_on='state_po',right_index=True)
emotional_health2 = emotional_health2.merge(single_parent_HH_pct_df,how='left',left_on='state_po',right_index=True)
emotional_health2 = emotional_health2.merge(pop65plus_df,how='left',left_on='state_po',right_index=True)
emotional_health2 = emotional_health2.merge(Obesity_pct_df,how='left',left_on='state_po',right_index=True)
emotional_health2 = emotional_health2.merge(rural_urban_df,how='left',left_on='state_po',right_index=True)

covidSt_df1.to_pickle("data/covidSt_df1.pkl")
emotional_health2.to_pickle("data/emotional_health2.pkl")
covidSt_df1 = pd.read_pickle("data/covidSt_df1.pkl")
covidSt_df1 = covidSt_df1.merge(state_region_xwalk,how='left',left_index=True,right_index=True)
covidSt_df1.loc[covidSt_df1.index=='DC','CensusRegionName'] = 'South'
covidSt_df1.loc[covidSt_df1.index=='DC','CensusDivisionName'] = 'South Atlantic'

covidSt_df1['pres_pct_D_Diff'] = covidSt_df1['pres_pct_D_2020']-covidSt_df1['pres_pct_D_2016']
covidSt_df1['pres_pct_R_Diff'] = covidSt_df1['pres_pct_R_2020']-covidSt_df1['pres_pct_R_2016']
covidSt_df1['pres_pct_Other_Diff'] = covidSt_df1['pres_pct_Other_2020']-covidSt_df1['pres_pct_Other_2016']
covidSt_df1['pres_pct_R_Diff_Diff'] = covidSt_df1['pres_pct_R_Diff']-covidSt_df1['pres_pct_D_Diff']
covidSt_df1.to_pickle("data/covidSt_df1.pkl")

"""## Counties (**county_df8** is the final product)

Cutoff at 11/2/2020 the nationwide state general elections were held 11/3/2020
"""

# covid = pd.read_csv('data/us-counties.csv')
# state_abbr = pd.read_csv('data/state_dict.csv')
covid = pd.read_pickle('data/covid2.pkl')
covid = covid[covid.date<'2020-11-03']

# county_fips = pd.read_pickle('data/county_fips.pkl')
# county_fips = covid[['ctySt','fips']].drop_duplicates()

# covid = covid.merge(state_abbr,how='inner',right_on='state',left_on='state') 
# covid = covid.dropna()
# covid.rename({'state_abbr':'state_po'},axis=1,inplace=True)
# covid['fips'] = covid['fips'].apply(np.int64)

# covid.to_pickle('data/covid2.pkl')
# county_fips.to_pickle('data/county_fips.pkl')

# pop_tot = {}
# pop_20_29 = {}
# pop_30_39 = {}
# pop_65_up = {}

county_demo = pd.read_csv("data/countypopmonthasrh.csv")
county_demo.rename({'state':'state_code', 'county':'fips'}, axis=1, inplace=True)
county_demo = county_demo.merge(state_abbr,how='inner',right_on='state',left_on='stname')
county_demo = county_demo.merge(county_fips,how='inner',right_on='fips',left_on='fips')
county_demo.rename({'state_abbr':'state_po'}, axis=1, inplace=True)
county_demo.shape #535230
#yearref=1 means 2010 census, agegrp=0 means total
pop_65_up = county_demo[(county_demo.yearref==1) & (county_demo.agegrp >13) & (county_demo.agegrp<19)][['fips','tot_pop']].drop_duplicates()
pop_65_up = pop_65_up.groupby('fips').sum()
pop_65_up.rename({'tot_pop':'pop_65_up'},axis=1,inplace=True)
pop_20_29 = county_demo[(county_demo.yearref==1) & ((county_demo.agegrp ==5) | (county_demo.agegrp==6))][['fips','tot_pop']].drop_duplicates()
pop_20_29 = pop_20_29.groupby('fips').sum()
pop_20_29.rename({'tot_pop':'pop_20_29'},axis=1,inplace=True)
pop_30_39 = county_demo[(county_demo.yearref==1) & ((county_demo.agegrp ==7) | (county_demo.agegrp==8))][['fips','tot_pop']].drop_duplicates()
pop_30_39 = pop_30_39.groupby('fips').sum()
pop_30_39.rename({'tot_pop':'pop_30_39'},axis=1,inplace=True)

county_demo = county_demo[(county_demo.yearref==1) & (county_demo.agegrp==0)]
county_demo = county_demo[['fips','tot_pop','tot_male','wa_male','wa_female',\
                           'ba_male','ba_female','h_male','h_female','hba_male','hba_female','aa_male','aa_female']]
county_demo.shape #(3130, 13)
county_demo = county_demo.merge(pop_20_29,how='inner',right_on='fips',left_on='fips')
county_demo = county_demo.merge(pop_30_39,how='inner',right_on='fips',left_on='fips')
county_demo = county_demo.merge(pop_65_up,how='inner',right_on='fips',left_on='fips')

county_demo.head()

# peaks_c = pd.read_pickle("data/v2/peaks_cases.pkl")
# widths_c = pd.read_pickle("data/v2/widths_cases.pkl")
# heights_c = pd.read_pickle("data/v2/heights_cases.pkl")
# no_peaks_c = pd.read_pickle("data/v2/no_peaks_cases.pkl")

# peaks_arr = peaks_c.T.to_dict('list')
# widths_arr = widths_c.T.to_dict('list')
# heights_arr = heights_c.T.to_dict('list')
# no_peaks = no_peaks_c.values.tolist()[0]

max_yaxis = 0
max_xaxis = 0
wl = 7

for fip in list(covid.fips.unique()):
    covid_plot = covid[covid.fips== fip]
    # covid_plot["daily_d"] = covid_plot.deaths_n.diff()
    covid_plot["daily_d"] = covid_plot.deaths.diff()
    # covid_plot["daily_c"] = covid_plot.cases_n.diff()
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
max_xaxis = 287
max_yaxis = 3609
wl = 7

peaks_arr = {}
widths_arr = {}
heights_arr = {}
areas_arr = {}
areas_KR_arr = {}
no_peaks_arr = {}

beta = 0.3
max_no_height =8

for fip in list(covid.fips.unique()):
    covid_plot = covid[covid.fips== fip]
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

### to construct dataframe to graph kernel regression charts by county
# Output from above
max_xaxis = 287
max_yaxis = 3609
wl = 7
y_ma_arr = {}
yHatk_arr = {}
beta = 0.3
max_no_height =8
covid = covid[covid.date<'2020-11-03']
# states_of_interest = ['NC']
states_of_interest = covid.State.unique()

for st in states_of_interest:
  temp = covid[covid.State== st]
  for ctySt in list(temp.ctySt.unique()):
      covid_plot = covid[covid.ctySt== ctySt]
      pad_zeros = max_xaxis - covid_plot.shape[0]
      zero_df = pd.DataFrame(0, index=np.arange(pad_zeros), columns=covid_plot.columns)
      covid_plot = pd.concat([zero_df,covid_plot])
      covid_plot["daily"] = covid_plot.cases_n.diff()        
      # covid_plot["daily"] = covid_plot.deaths_n.diff()
      covid_plot.loc[covid_plot.daily<0,'daily'] = 0 #try this in lieu of above line
      covid_plot["ma"] = np.rint(covid_plot.daily.rolling(window=wl).mean())
      covid_plot = covid_plot.iloc[wl:,:]
      covid_plot = covid_plot.reset_index(drop=True)
      covid_plot['periods'] = covid_plot.index        
      xTrain = covid_plot.periods.to_numpy()[:,np.newaxis]
      yTrain = covid_plot.ma
      h = beta*median_distance(xTrain)
      yHatk = kernel_regression_fit_and_predict(xTrain, yTrain, xTrain, h, beta)
      yHatk_arr[ctySt] = yHatk
      y_ma_arr[ctySt] = yTrain

      # fig, ax = plt.subplots( nrows=1, ncols=1, figsize=(30,16) )
      # ax.plot(xTrain,yTrain)
      # ax.axis('off')
      # fig.savefig(directory+ctySt+'.png')
      # plt.close(fig)
      # fig.clear()
      # fig, ax = plt.subplots( nrows=1, ncols=1, figsize=(30,16) )
      # ax.plot(xTrain,yHatk)
      # ax.axis('off')
      # fig.savefig(directory+ctySt+'_KR.png')
      # fig.clear()
      # plt.close(fig)
      # print(str(len(heights_arr)) +' ' + ctySt)
      if len(yHatk_arr) % 50 == 0:
        gc.collect()

y_ma_c = pd.DataFrame.from_dict(y_ma_arr).T
yHatk_c = pd.DataFrame.from_dict(yHatk_arr).T

# ctySt = 'Mecklenburg_NC'
# sns.lineplot(x=y_ma_c.columns, y=y_ma_c.loc[ctySt,:])
# sns.lineplot(x=yHatk_c.columns, y=y_ma_c.loc[ctySt,:])

# peaks_c = pd.DataFrame.from_dict(peaks_arr).T
# widths_c = pd.DataFrame.from_dict(widths_arr).T
# heights_c = pd.DataFrame.from_dict(heights_arr).T
# areas_c = pd.DataFrame.from_dict(areas_arr).T
# areas_KR_c = pd.DataFrame.from_dict(areas_KR_arr).T
# no_peaks_c   = pd.DataFrame(no_peaks_arr, index=[0]).T

# peaks_c.to_pickle("data/v2/peaks_cases.pkl")
# widths_c.to_pickle("data/v2/widths_cases.pkl")
# heights_c.to_pickle("data/v2/heights_cases.pkl")
# areas_c.to_pickle("data/v2/areas_cases.pkl")
# areas_KR_c.to_pickle("data/v2/areas_KR_cases.pkl")
# no_peaks_c.to_pickle("data/v2/no_peaks_cases.pkl")

peaks_d = pd.DataFrame.from_dict(peaks_arr).T
widths_d = pd.DataFrame.from_dict(widths_arr).T
heights_d = pd.DataFrame.from_dict(heights_arr).T
areas_d = pd.DataFrame.from_dict(areas_arr).T
areas_KR_d = pd.DataFrame.from_dict(areas_KR_arr).T
no_peaks_d = pd.DataFrame(no_peaks_arr, index=[0]).T

peaks_d.to_pickle("data/v2/peaks_deaths.pkl")
widths_d.to_pickle("data/v2/widths_deaths.pkl")
heights_d.to_pickle("data/v2/heights_deaths.pkl")
areas_d.to_pickle("data/v2/areas_deaths.pkl")
areas_KR_d.to_pickle("data/v2/areas_KR_deaths.pkl")
no_peaks_d.to_pickle("data/v2/no_peaks_deaths.pkl")

# Cases
# peaks_c.rename({0:'peak1_c',1:'peak2_c',2:'peak3_c',3:'peak4_c',4:'peak5_c',5:'peak6_c',6:'peak7_c',7:'peak8_c'}, axis=1, inplace=True)
# widths_c.rename({0:'width1_c',1:'width2_c',2:'width3_c',3:'width4_c',4:'width5_c',5:'width6_c',6:'width7_c',7:'width8_c'}, axis=1, inplace=True)
# heights_c.rename({0:'height1_c',1:'height2_c',2:'height3_c',3:'height4_c',4:'height5_c',5:'height6_c',6:'height7_c',7:'height8_c'}, axis=1, inplace=True)
# areas_c.rename({0:'area1_c',1:'area2_c',2:'area3_c',3:'area4_c',4:'area5_c',5:'area6_c',6:'area7_c',7:'area8_c'}, axis=1, inplace=True)
# areas_KR_c.rename({0:'area1_KR_c',1:'area2_KR_c',2:'area3_KR_c',3:'area4_KR_c',4:'area5_KR_c',5:'area6_KR_c',6:'area7_KR_c',7:'area8_KR_c'}, axis=1, inplace=True)
# no_peaks_c.rename({0:'no_peaks_c'}, axis=1, inplace=True)

# peaks_c.to_pickle("data/v2/peaks_cases.pkl")
# widths_c.to_pickle("data/v2/widths_cases.pkl")
# heights_c.to_pickle("data/v2/heights_cases.pkl")
# areas_c.to_pickle("data/v2/areas_cases.pkl")
# areas_KR_c.to_pickle("data/v2/areas_KR_cases.pkl")
# no_peaks_c.to_pickle("data/v2/no_peaks_cases.pkl")

# Death counts
peaks_d.rename({0:'peak1_d',1:'peak2_d',2:'peak3_d',3:'peak4_d',4:'peak5_d',5:'peak6_d',6:'peak7_d',7:'peak8_d'}, axis=1, inplace=True)
widths_d.rename({0:'width1_d',1:'width2_d',2:'width3_d',3:'width4_d',4:'width5_d',5:'width6_d',6:'width7_d',7:'width8_d'}, axis=1, inplace=True)
heights_d.rename({0:'height1_d',1:'height2_d',2:'height3_d',3:'height4_d',4:'height5_d',5:'height6_d',6:'height7_d',7:'height8_d'}, axis=1, inplace=True)
areas_d.rename({0:'area1_d',1:'area2_d',2:'area3_d',3:'area4_d',4:'area5_d',5:'area6_d',6:'area7_d',7:'area8_d'}, axis=1, inplace=True)
areas_KR_d.rename({0:'area1_KR_d',1:'area2_KR_d',2:'area3_KR_d',3:'area4_KR_d',4:'area5_KR_d',5:'area6_KR_d',6:'area7_KR_d',7:'area8_KR_d'}, axis=1, inplace=True)
no_peaks_d.rename({0:'no_peaks_d'}, axis=1, inplace=True)

peaks_d.to_pickle("data/v2/peaks_deaths.pkl")
widths_d.to_pickle("data/v2/widths_deaths.pkl")
heights_d.to_pickle("data/v2/heights_deaths.pkl")
areas_d.to_pickle("data/v2/areas_deaths.pkl")
areas_KR_d.to_pickle("data/v2/areas_KR_deaths.pkl")
no_peaks_d.to_pickle("data/v2/no_peaks_deaths.pkl")

county_df1[county_df1.index==705] #should be empty

county_demo.set_index('fips',inplace=True)
county_demo.head()

county_df1 = county_demo.merge(peaks_c,how='inner',right_index=True,left_index=True)
county_df1 = county_df1.merge(widths_c,how='inner',right_index=True,left_index=True)
county_df1 = county_df1.merge(heights_c,how='inner',right_index=True,left_index=True)
county_df1 = county_df1.merge(areas_c,how='inner',right_index=True,left_index=True)
county_df1 = county_df1.merge(areas_KR_c,how='inner',right_index=True,left_index=True)
county_df1 = county_df1.merge(no_peaks_c,how='inner',right_index=True,left_index=True)

county_df1 = county_df1.merge(peaks_d,how='inner',right_index=True,left_index=True)
county_df1 = county_df1.merge(widths_d,how='inner',right_index=True,left_index=True)
county_df1 = county_df1.merge(heights_d,how='inner',right_index=True,left_index=True)
county_df1 = county_df1.merge(areas_d,how='inner',right_index=True,left_index=True)
county_df1 = county_df1.merge(areas_KR_d,how='inner',right_index=True,left_index=True)
county_df1 = county_df1.merge(no_peaks_d,how='inner',right_index=True,left_index=True)

county_df1.head()

binyu_nh = pd.read_csv('data/binyu_nh.csv')
county_df1 = county_df1.merge(binyu_nh,how='inner',left_index=True,right_on='countyFIPS')
county_df1.to_pickle('data/county_df1.pkl')

no_counties = county_df1.shape[0]
temp1=np.zeros(no_counties)
temp2=np.zeros(no_counties)
temp3=np.zeros(no_counties)
temp4=np.zeros(no_counties)
temp5=np.zeros(no_counties)
temp6=np.zeros(no_counties)
temp7=np.zeros(no_counties)
temp8=np.zeros(no_counties)
temp = [0,0,0,0,0,0,0,0]

rural = list(county_df1.columns.tolist()).index('Rural-UrbanContinuumCode2013')
popfmle = list(county_df1.columns.tolist()).index('PopFmle>842010')+1
idx3 = list(county_df1.columns.tolist()).index('3-YrMortalityAge15-24Years2015-17')
idx4 = list(county_df1.columns.tolist()).index('mortality2015-17Estimated')
idx5 = list(county_df1.columns.tolist()).index('% Uninsured')
idx6 = list(county_df1.columns.tolist()).index('Social Association Rate')+1
socioecon = list(county_df1.columns[rural:popfmle].append(county_df1.columns[idx3:idx4]).append(county_df1.columns[idx5:idx6])) #includes dem_to_rep_ratio
columns_to_drop = county_df1.columns[county_df1.isna().sum()/county_df1.shape[0]>0.0]
columns_to_drop = [*columns_to_drop,*['countyFIPS','ctySt']]
socioecon_df = county_df1.drop(columns_to_drop,axis=1)
socioecon = [elem for elem in socioecon if elem not in columns_to_drop]
# socioecon = [*socioecon,*['MBdim_c','MBdimKR_c','MBdim_d','MBdimKR_d']]
socioecon_df = socioecon_df.loc[:,socioecon]
# socioecon_df = socioecon_df.set_index('ctySt',drop=False)

county_df2 = socioecon_df.merge(covid_df1,how='inner',right_index=True,left_index=True)

county_df1.head()

for i in range(county_df1.shape[0]):    
    x = county_df1[['peak1_c','peak2_c','peak3_c','peak4_c','peak5_c','peak6_c','peak7_c','peak8_c']][county_df1[['peak1_c','peak2_c','peak3_c','peak4_c','peak5_c','peak6_c','peak7_c','peak8_c']] > 0].iloc[i,:].nsmallest(8)
    temp[:len(x)] = x
    temp1[i] = temp[0]
    temp2[i] = temp[1]
    temp3[i] = temp[2]
    temp4[i] = temp[3]
    temp5[i] = temp[4]
    temp6[i] = temp[5]
    temp7[i] = temp[6]
    temp8[i] = temp[7]
    temp = [0,0,0,0,0,0,0,0]
    
county_df1.loc[:,'localmax1_c'] = temp1
county_df1.loc[:,'localmax2_c'] = temp2
county_df1.loc[:,'localmax3_c'] = temp3
county_df1.loc[:,'localmax4_c'] = temp4
county_df1.loc[:,'localmax5_c'] = temp5
county_df1.loc[:,'localmax6_c'] = temp6
county_df1.loc[:,'localmax7_c'] = temp7
county_df1.loc[:,'localmax8_c'] = temp8

for i in range(county_df1.shape[0]):
    x = county_df1[['peak1_d','peak2_d','peak3_d','peak4_d','peak5_d','peak6_d','peak7_d','peak8_d']][county_df1[['peak1_d','peak2_d','peak3_d','peak4_d','peak5_d','peak6_d','peak7_d','peak8_d']] > 0].iloc[i,:].nsmallest(8)
    temp[:len(x)] = x
    temp1[i] = temp[0]
    temp2[i] = temp[1]
    temp3[i] = temp[2]
    temp4[i] = temp[3]
    temp5[i] = temp[4]
    temp6[i] = temp[5]
    temp7[i] = temp[6]
    temp8[i] = temp[7]
    temp = [0,0,0,0,0,0,0,0]

county_df1.loc[:,'localmax1_d'] = temp1
county_df1.loc[:,'localmax2_d'] = temp2
county_df1.loc[:,'localmax3_d'] = temp3
county_df1.loc[:,'localmax4_d'] = temp4
county_df1.loc[:,'localmax5_d'] = temp5
county_df1.loc[:,'localmax6_d'] = temp6
county_df1.loc[:,'localmax7_d'] = temp7
county_df1.loc[:,'localmax8_d'] = temp8

"""## Transform peaks to distance from Nov. 3 2020 election"""

peaks = ['peak1_c','peak2_c','peak3_c','peak4_c',\
         'peak5_c','peak6_c','peak7_c','peak8_c',\
         'peak1_d','peak2_d','peak3_d','peak4_d',\
         'peak5_d','peak6_d','peak7_d','peak8_d',\
         'localmax1_c', 'localmax2_c', 'localmax3_c', 'localmax4_c',\
         'localmax5_c', 'localmax6_c', 'localmax7_c', 'localmax8_c',\
         'localmax1_d', 'localmax2_d', 'localmax3_d', 'localmax4_d',\
         'localmax5_d', 'localmax6_d', 'localmax7_d', 'localmax8_d']
for peak in peaks:
  new_col = peak + '_dist'
  # 305 days from Jan. 1 until Nov. 2, 2020, and 1/21/2020 is min of covid.date
  county_df1.loc[:,new_col] = 305 - county_df1.loc[:,peak] - 21

county_df1.to_pickle("data/county_df1.pkl")

"""### States: Calculate exact area"""

county_df8 = county_df8.loc[:,~county_df8.columns.duplicated()]
county_df8.to_pickle("county_df8.pkl")

"""# Load Data

## States
"""

# pd.set_option("display.max_rows", None, "display.max_columns", None)
covidSt_df1= pd.read_pickle("data/covidSt_df1.pkl")
emotional_health2= pd.read_pickle("data/emotional_health2.pkl")
state_region_xwalk = pd.read_pickle('data/state_region_xwalk.pkl')

"""## Counties"""

# county_df8 = pd.read_pickle("data/county_df8.pkl")
# pd.set_option("display.max_rows", None, "display.max_columns", None)

# no_peaks is erroneous. For county 48033, e.g., all widths and peaks are 0, but has no_peaks = 7
idx1 = (county_df8.columns.tolist()).index('peak1_c')
idx2 = (county_df8.columns.tolist()).index('peak8_c')+1

for row in range(county_df8.shape[0]):
  county_df8.loc[row,'no_peaks_c'] = np.sum(county_df8.iloc[row,idx1:idx2]>0)

idx1 = (county_df8.columns.tolist()).index('peak1_d')
idx2 = (county_df8.columns.tolist()).index('peak8_d')+1
for row in range(county_df8.shape[0]):
  county_df8.loc[row,'no_peaks_d'] = np.sum(county_df8.iloc[row,idx1:idx2]>0)

# county_df1 = pd.read_pickle("data/county_df1.pkl")
county_df6 = pd.read_pickle("data/county_df6.pkl")

"""# State-Level Viz of Region and Delta-Rep/Dep"""

# covidSt_df1.columns.to_list()
# temp = covidSt_df1[['pres_pct_R_Diff','pres_pct_D_Diff','pres_pct_Other_Diff','CensusRegionName']]
# varname='pres_pct_R_Diff'
varname='pres_pct_D_Diff'
# varname='pres_pct_Other_Diff'
temp=temp.sort_values(by=varname,ascending=False)
# temp=temp.sort_values(by=varname,ascending=True)
plt.rcParams["figure.figsize"] = [15,7]
plt.rcParams["figure.autolayout"] = True
bar_plot = sns.barplot(x=temp.index,y=temp[varname], hue=temp['CensusRegionName'])
plt.xticks(rotation=45)
plt.show()
sns.displot(covidSt_df1, x='pres_pct_D_Diff',hue="CensusRegionName", kind="kde",  fill=True)
sns.displot(covidSt_df1, x='pres_pct_Other_Diff',hue="CensusRegionName", kind="kde",  fill=True)
sns.displot(covidSt_df1, x='pres_pct_R_Diff',hue="CensusRegionName", kind="kde",  fill=True)
sns.displot(covidSt_df1, x='pres_pct_R_Diff_Diff',hue="CensusRegionName", kind="kde",  fill=True)

mean(west.pres_pct_D_Diff)
mean(south.pres_pct_D_Diff)

northeast.pres_pct_D_Diff.describe()
south.pres_pct_D_Diff.describe()

covidSt_df1.loc[covidSt_df1.index=='UT','pres_pct_Other_2020']
covidSt_df1['pres_pct_Other_Diff'].describe()
covidSt_df1.loc[covidSt_df1.index=='UT',['pres_pct_D_2020','pres_pct_R_2020','pres_pct_Other_2020']]

"""# Association between Anxiety/Depression and Peaks (NCHS Household Pulse Survey)

## US Senate
"""

senate_2016 = pd.read_csv('data/2016-precinct-senate-NH.tab', sep='\t', lineterminator='\r')

"""## State-level

### Setup
"""

nchs = pd.read_csv('data/Indicators_of_Anxiety_or_Depression_Based_on_Reported_Frequency_of_Symptoms_During_Last_7_Days.csv')
state_abbr = pd.read_csv('data/state_dict.csv')
state_demo = pd.read_csv("data/DECENNIALPL2020.P1_data_with_overlays_2022-01-19T210012.csv")
emotional_health2 = pd.read_pickle("data/emotional_health2.pkl")  #latest emotional_health 2/26/2022

state_demo = state_demo.merge(state_abbr,how='inner',right_on='state',left_on='NAME')
state_demo.rename({'state_abbr':'state_po','P1_001N':'total_pop','P1_003N':'white','P1_004N':'black','P1_006N':'asian'},axis=1,inplace=True)
state_demo = state_demo[['state_po','total_pop','white','black','asian']]
state_demo.set_index('state_po',drop=True,inplace=True)
state_demo['total_pop'] = pd.to_numeric(state_demo['total_pop'])
state_demo['white'] = pd.to_numeric(state_demo['white'])/state_demo['total_pop']
state_demo['black'] = pd.to_numeric(state_demo['black'])/state_demo['total_pop']
state_demo['asian'] = pd.to_numeric(state_demo['asian'])/state_demo['total_pop']

emotional_health2 = emotional_health2.merge(state_demo,how='inner',right_index=True,left_on='state_po')
# emotional_health2.to_pickle("data/emotional_health2.pkl")

# emotional_health = nchs.merge(state_abbr,how='inner',right_on='state',left_on='State')
# emotional_health.rename({'Time Period Start Date':'start_date','Time Period End Date':'end_date',\
#                          'state_abbr':'state_po','Time Period':'time_period'}, axis=1, inplace=True)
# emotional_health = emotional_health[emotional_health.time_period <= 18]
# # emotional_health = emotional_health.drop(columns=['index'])
# emotional_health.reset_index(drop=True,inplace=True)
# emotional_health.loc[:,'start_int'] = [(datetime.datetime.strptime(date,"%m/%d/%Y").date()-min_date).days for date in emotional_health.loc[:,'start_date']]
# emotional_health.loc[:,'end_int'] = [(datetime.datetime.strptime(date,"%m/%d/%Y").date()-min_date).days for date in emotional_health.loc[:,'end_date']]

anxiety_depression = emotional_health2[emotional_health2.Indicator == 'Symptoms of Anxiety Disorder or Depressive Disorder']
anxiety = emotional_health2[emotional_health2.Indicator == 'Symptoms of Anxiety Disorder']
depression = emotional_health2[emotional_health2.Indicator == 'Symptoms of Depressive Disorder']

anxiety_depression['Value_1'] = anxiety_depression['Value']
anxiety['Value_1'] = anxiety['Value']
depression['Value_1'] = depression['Value']

for state_po in list(anxiety_depression.state_po.unique()):
  temp = anxiety_depression[anxiety_depression.state_po== state_po]
  for time_period in list(temp.time_period.unique()):
    if time_period == 1:
      anxiety_depression.loc[(anxiety_depression.state_po==state_po) & (anxiety_depression.time_period==1),'Value_1'] = 0
    else:
      anxiety_depression.loc[(anxiety_depression.state_po==state_po) & (anxiety_depression.time_period==time_period),'Value_1'] = \
      float(anxiety_depression.loc[(anxiety_depression.state_po==state_po) & (anxiety_depression.time_period==time_period-1),'Value'])

for state_po in list(anxiety.state_po.unique()):
  temp = anxiety[anxiety.state_po== state_po]
  for time_period in list(temp.time_period.unique()):
    if time_period == 1:
      anxiety.loc[(anxiety.state_po==state_po) & (anxiety.time_period==1),'Value_1'] = 0
    else:
      anxiety.loc[(anxiety.state_po==state_po) & (anxiety.time_period==time_period),'Value_1'] = \
      float(anxiety.loc[(anxiety.state_po==state_po) & (anxiety.time_period==time_period-1),'Value'])

for state_po in list(depression.state_po.unique()):
  temp = depression[depression.state_po== state_po]
  for time_period in list(temp.time_period.unique()):
    if time_period == 1:
      depression.loc[(depression.state_po==state_po) & (depression.time_period==1),'Value_1'] = 0
    else:
      depression.loc[(depression.state_po==state_po) & (depression.time_period==time_period),'Value_1'] = \
      float(depression.loc[(depression.state_po==state_po) & (depression.time_period==time_period-1),'Value'])

# anxiety_depression.to_pickle("data/anxiety_depression.pkl")
# anxiety.to_pickle("data/anxiety.pkl")
# depression.to_pickle("data/depression.pkl")

anxiety_depression = pd.read_pickle("data/anxiety_depression.pkl")
anxiety   = pd.read_pickle("data/anxiety.pkl")
depression = pd.read_pickle("data/depression.pkl")

min_date = datetime.datetime.strptime("1/22/2020","%m/%d/%Y").date()
datetime.datetime.strptime('2020-01-29',"%Y-%m-%d")-datetime.datetime.strptime(min(covidSt.date),"%Y-%m-%d")
(datetime.datetime.strptime('2020-01-29',"%Y-%m-%d")-pd.to_datetime(min(covidSt.date))).days
pd.to_datetime(min(covidSt.date))+datetime.timedelta(days=10)

case_peaks = ['peak1_c', 'peak2_c', 'peak3_c', 'peak4_c', 'peak5_c', 'peak6_c', 'peak7_c', 'peak8_c','no_peaks_c']
case_areas = ['area1_c','area2_c','area3_c','area4_c','area5_c','area6_c','area7_c','area8_c',
              'area1_KR_c','area2_KR_c','area3_KR_c','area4_KR_c','area5_KR_c','area6_KR_c','area7_KR_c','area8_KR_c']
death_peaks = ['peak1_d', 'peak2_d', 'peak3_d', 'peak4_d', 'peak5_d', 'peak6_d', 'peak7_d', 'peak8_d','no_peaks_d']
death_areas = ['area1_d','area2_d','area3_d','area4_d','area5_d','area6_d','area7_d','area8_d',
              'area1_KR_d','area2_KR_d','area3_KR_d','area4_KR_d','area5_KR_d','area6_KR_d','area7_KR_d','area8_KR_d']

nearest_peak_c = {}
nearest_area_c = {}  
no_peaks_c = {}
nearest_peak_d = {}
nearest_area_d = {}
no_peaks_d = {}

for row in emotional_health.index:
  st = emotional_health.loc[row,'state_po']
  no_peaks_c[row] = 0
  no_peaks_d[row] = 0
  for i,peak in enumerate(case_peaks):
    if (emotional_health.loc[row,'start_int'] > covidSt_df1.loc[st,peak]):
      no_peaks_c[row] = no_peaks_c[row] + 1
      if row not in nearest_peak_c.keys():
        nearest_peak_c[row] = covidSt_df1.loc[st,peak]
        nearest_area_c[row] = covidSt_df1.loc[st,case_areas[i]]
  for i,peak in enumerate(death_peaks):
    if (emotional_health.loc[row,'start_int'] > covidSt_df1.loc[st,peak]):
      no_peaks_d[row] = no_peaks_d[row] + 1
      if row not in nearest_peak_d.keys():
        nearest_peak_d[row] = covidSt_df1.loc[st,peak]
        nearest_area_d[row] = covidSt_df1.loc[st,death_areas[i]]

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

emotional_health = emotional_health.merge(nearest_peak_c_df,how='left',right_index=True,left_index=True)
emotional_health = emotional_health.merge(nearest_area_c_df,how='left',right_index=True,left_index=True)
emotional_health = emotional_health.merge(no_peaks_c_df,how='left',right_index=True,left_index=True)
emotional_health = emotional_health.merge(nearest_peak_d_df,how='left',right_index=True,left_index=True)
emotional_health = emotional_health.merge(nearest_area_d_df,how='left',right_index=True,left_index=True)
emotional_health = emotional_health.merge(no_peaks_d_df,how='left',right_index=True,left_index=True)

cols = ['peak1_c','peak2_c','peak3_c','peak1_d','peak2_d','peak3_d','area1_c','area2_c','area3_c','area1_d','area2_d','area3_d']
emotional_health = emotional_health.merge(covidSt_df1[cols],how='left',right_index=True,left_on='state_po')

covidSt_df1 = covidSt_df1.merge(state_region_xwalk,how='left',right_index=True,left_index=True)
covidSt_df1.head()
# covidSt_df1.to_pickle('data/covidSt_df1')

temp = covidSt_df1[covidSt_df1.CensusRegionName=='West']
temp = covidSt_df1[covidSt_df1.CensusRegionName=='Midwest']
temp = covidSt_df1[covidSt_df1.CensusRegionName=='South']
temp = covidSt_df1[covidSt_df1.CensusRegionName=='Northeast']
temp.no_peaks_d.describe()

"""### Transform peak data to number of days until the election day"""

# emotional_health2.to_pickle("data/emotional_health2.pkl")

peaks = ['nearest_peak_c', 'nearest_peak_d']
for peak in peaks:
  new_col = peak + '_dist'
  # 305 days from Jan. 1 until Nov. 2, 2020, and 1/22/2020 is min of covid.date
  emotional_health.loc[:,new_col] = 305 - emotional_health.loc[:,peak] - 22

# emotional_health.to_pickle("data/emotional_health.pkl")

# to normalize or not to normalize?
emotional_health.loc[:,'nearest_peak_c_dist_std'] = emotional_health.loc[:,'nearest_peak_c_dist']/np.linalg.norm(emotional_health.loc[:,'nearest_peak_c_dist'])
emotional_health.loc[:,'nearest_peak_d_dist_std'] = emotional_health.loc[:,'nearest_peak_d_dist']/np.linalg.norm(emotional_health.loc[:,'nearest_peak_d_dist'])
emotional_health.loc[:,'nearest_area_c_std'] = emotional_health.loc[:,'nearest_area_c']/np.linalg.norm(emotional_health.loc[:,'nearest_area_c'])
emotional_health.loc[:,'nearest_area_d_std'] = emotional_health.loc[:,'nearest_area_d']/np.linalg.norm(emotional_health.loc[:,'nearest_area_d'])

# add 2016 winning party info for all states
# covidSt_df1.set_index('state_po',drop=True,inplace=True)
emotional_health = emotional_health.merge(covidSt_df1.loc[:,['pres_winner_2016']],how='left',right_index=True,left_on='state_po')

vars = ['FracMale2017','MedianAge2010','HeartDiseaseMortality','PopulationEstimate2018',\
        'Smokers_Percentage','RespMortalityRate2014','DiabetesPercentage','unemployed_pct',\
        'single_parent_HH_pct','pop65plus','Obesity_pct']


    
    
###########################################################################
############### for Journal of Politics JOP submission ####################
###########################################################################

    ## Symptoms of Anxiety Disorder or Depressive Disorder

# data = anxiety_depression
data = anxiety_depression[anxiety_depression.pres_winner_2016 == 'REPUBLICAN']
data = anxiety_depression[anxiety_depression.pres_winner_2016 == 'DEMOCRAT']
# data = anxiety_depression[anxiety_depression.rural_urban <= 3]
# data = anxiety_depression[anxiety_depression.rural_urban > 3]

# Y, X = dmatrices('Value ~ Value_1 + time_period + State + FracMale2017 + MedianAge2010 + HeartDiseaseMortality +\
# Smokers_Percentage + RespMortalityRate2014 + DiabetesPercentage + unemployed_pct + single_parent_HH_pct +\
# pop65plus + Obesity_pct + rural_urban + white + black + asian + \
# nearest_peak_c + nearest_area_c + no_peaks_c + nearest_peak_d + nearest_area_d + no_peaks_d', data, return_type="dataframe")

Y, X = dmatrices('Value ~ Value_1 + time_period + State + FracMale2017 + MedianAge2010 + HeartDiseaseMortality +\
Smokers_Percentage + RespMortalityRate2014 + DiabetesPercentage + unemployed_pct + single_parent_HH_pct +\
pop65plus + Obesity_pct + rural_urban + white + black + asian + \
nearest_peak_c + nearest_peak_d', data, return_type="dataframe")

Y = np.ravel(Y)

model0 = sm.OLS(Y,X.astype(float))
results0 = model0.fit()

coefs0 = pd.DataFrame({
    'coef': results0.params.values,
    'stderr':results0.bse,
    'pvalue': results0.pvalues
    }).sort_values(by='pvalue', ascending=True)
coefs0.loc['time_period',:]
coefs0.loc['Value_1',:]
coefs0.loc['nearest_peak_c',:]
coefs0.loc['nearest_peak_d',:]
# print(results0.summary())  #data.shape

## Symptoms of Anxiety Disorder

# data = anxiety
data = anxiety[anxiety.pres_winner_2016 == 'REPUBLICAN']
data = anxiety[anxiety.pres_winner_2016 == 'DEMOCRAT']
# data = anxiety[anxiety.rural_urban <= 3]
# data = anxiety[anxiety.rural_urban > 3]

# Y, X = dmatrices('Value ~ Value_1 + time_period + State + FracMale2017 + MedianAge2010 + HeartDiseaseMortality +\
# Smokers_Percentage + RespMortalityRate2014 + DiabetesPercentage + unemployed_pct + single_parent_HH_pct +\
# pop65plus + Obesity_pct + rural_urban + white + black + asian + \
# nearest_peak_c + nearest_area_c + no_peaks_c + nearest_peak_d + nearest_area_d + no_peaks_d', data, return_type="dataframe")

Y, X = dmatrices('Value ~ Value_1 + time_period + State + FracMale2017 + MedianAge2010 + HeartDiseaseMortality +\
Smokers_Percentage + RespMortalityRate2014 + DiabetesPercentage + unemployed_pct + single_parent_HH_pct +\
pop65plus + Obesity_pct + rural_urban + white + black + asian + nearest_peak_c', data, return_type="dataframe")


Y = np.ravel(Y)

model0 = sm.OLS(Y,X.astype(float))
results0 = model0.fit()

coefs0 = pd.DataFrame({
    'coef': results0.params.values,
    'stderr':results0.bse,
    'pvalue': results0.pvalues
    }).sort_values(by='pvalue', ascending=True)
coefs0.loc['time_period',:]
coefs0.loc['Value_1',:]
coefs0.loc['nearest_peak_c',:]
# coefs0.loc['nearest_peak_d',:]

## Symptoms of Depressive Disorder

# data = depression
# data = depression[depression.pres_winner_2016 == 'REPUBLICAN']
# data = depression[depression.pres_winner_2016 == 'DEMOCRAT']
# data = depression[depression.rural_urban <= 3]
data = depression[depression.rural_urban > 3]

Y, X = dmatrices('Value ~ Value_1 + time_period + State + FracMale2017 + MedianAge2010 + HeartDiseaseMortality +\
Smokers_Percentage + RespMortalityRate2014 + DiabetesPercentage + unemployed_pct + single_parent_HH_pct +\
pop65plus + Obesity_pct + rural_urban + white + black + asian + \
nearest_peak_c + nearest_area_c + no_peaks_c + nearest_peak_d + nearest_area_d + no_peaks_d', data, return_type="dataframe")
Y = np.ravel(Y)

model0 = sm.OLS(Y,X.astype(float))
results0 = model0.fit()

coefs0 = pd.DataFrame({
    'coef': results0.params.values,
    'stderr':results0.bse,
    'pvalue': results0.pvalues
    }).sort_values(by='pvalue', ascending=True)
coefs0

# data = emotional_health[emotional_health.Indicator == 'Symptoms of Anxiety Disorder or Depressive Disorder']
# data = emotional_health[emotional_health.Indicator == 'Symptoms of Anxiety Disorder']
data = emotional_health[emotional_health.Indicator == 'Symptoms of Depressive Disorder']
data = data[data.pres_winner_2016 == 'DEMOCRAT']
# data = data[data.pres_winner_2016 == 'REPUBLICAN']

Y, X = dmatrices('Value ~ time_period + State + nearest_peak_c_dist + nearest_area_c + no_peaks_c +\
 nearest_peak_d_dist + nearest_area_d + no_peaks_d', data, return_type="dataframe")
Y = np.ravel(Y)

model0 = sm.OLS(Y,X.astype(float))
results0 = model0.fit()

coefs0 = pd.DataFrame({
    'coef': results0.params.values,
    'stderr':results0.bse,
    'pvalue': results0.pvalues
    }).sort_values(by='pvalue', ascending=True)
coefs0

# sns.lmplot(x="no_peaks_c", y="Value", hue="pres_winner_2016", data=anxiety_depression);
sns.lmplot(x="nearest_peak_d_dist", y="Value", data=anxiety_depression)