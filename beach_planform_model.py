# -*- coding: utf-8 -*-
"""
Created on Thu May 26 15:15:12 2022

@author: Rui  & Ana
"""

import SEMGrid
import matplotlib.pyplot as plt
import numpy as np
import wavetimeseries as wts
import geopandas as gpd
import pandas as pd
from scipy.optimize import minimize
import pickle
from shapely import geometry


def wave_builder(wave_inp, param = ''):
    """ create a wave time series from wave data
    param ->
      '' - total sea state
      '_sw' - swell waves
      '_ww' - wind waves
      '_sw', '_ww' - swell and wind waves
      '_sw_p1','_sw_p2', '_sw_p3', '_ww' - swell partitions 1 to 3 and wind waves
    """
    
    dir_param = []
    h_param = []
    t_param = []
    for param in param_lst: 
        dir_param.append(['Dir' + param])
        h_param.append(['Hs' + param])
        t_param.append(['Tm' + param]) # Tm 
        # t_param.append(['Tp' + param]) # for sensitive analisys with Tp - valid only to the total sea state
    wDir = []
    wT  = []
    wHs = []
    for dirp, tp, hp in zip(dir_param, t_param, h_param):
        wDir.append(waves.wave_data[dirp])
        wT.append(waves.wave_data[tp])
        wHs.append(waves.wave_data[hp])
    
    wDir = np.array(wDir).flatten()
    wT = np.array(wT).flatten()
    wHs = np.array(wHs).flatten()
    
   
    return wDir, wHs, wT


def wave_mean_dir(dir, weight = 1):
    """ compute mean wave dir
        if weight is equal to wave power compute mean power direction
    """
    x = np.sum(weight * np.cos(np.radians(dir)))
    y = np.sum(weight * np.sin(np.radians(dir)))
    dir_m = np.arctan2(y, x) * 180 / np.pi
    if dir_m < 0:
        dir_m += 360
    std_m = np.degrees(np.sqrt(-2 * np.log(np.sqrt(x**2 + y**2) / weight.sum()))) 
    return dir_m, std_m 

def rms(fixed_point_upwave, wDir, wT, wHs, fixed_point_downwave, linha_costa, node_spacing, ocean_side):
    x, y, swpr, mean_waves_mask = shoreline_builder(fixed_point_upwave, wDir, wT, wHs, fixed_point_downwave, linha_costa, node_spacing, ocean_side = ocean_side)
    dx = linha_costa.x - x[:, None]
    dy = linha_costa.y - y[:, None]
    rms = np.sqrt(np.sum(np.linalg.norm([dx, dy], axis=0)**2)/linha_costa.num_cells)
    print('rms -> ', rms)
    return rms

def filter_angles_between_azimuths(angles, az1, az2, ocean_side):
    if az1 < az2:
        mask = ((angles >= az1) & (angles <= az2))
    else:
        if ocean_side == 'left':
            mask = ((angles >= az1) | (angles <= az2)) # Ponta Negra # ANA
        else:
            mask = ((angles >= az2) & (angles <= az1)) # Puerto Huarmey; Agraria e Paraiso # ANA
                
    return mask


def shoreline_builder(fixed_point_upwave, wDir, wT, wHs, fixed_point_downwave,
                     linha_costa, node_spacing, ocean_side = 'right'):
    n_points = linha_costa.num_cells 
    x = np.zeros(n_points+1)
    y = np.zeros(n_points+1)
    x[0] = linha_costa.x[0,0]
    y[0] = linha_costa.y[0,0]
    
    wave_power = 1025 * 9.81**2 * wHs**2 * wT / (64 * np.pi)
    shelter_wave_power_ratio =  np.zeros(n_points+1)
    n_mask = np.empty(n_points+1) # A 
    mean_wDir_mask = np.empty(n_points+1) # A 
    mean_wT_mask = np.empty(n_points+1) #A
    mean_wHs_mask = np.empty(n_points+1) #A
    mean_stdDir_mask = np.empty(n_points+1) #A
            
    for i_point in range(n_points):
        dx = fixed_point_upwave[0] - x[i_point]
        dy = fixed_point_upwave[1] - y[i_point]
        azimuth_upwave = (np.mod(450 - np.rad2deg(np.arctan2(dy, dx)), 360).tolist())
        dx = fixed_point_downwave[0] - x[i_point]
        dy = fixed_point_downwave[1] - y[i_point]
        azimuth_downwave = (np.mod(450 - np.rad2deg(np.arctan2(dy, dx)), 360).tolist())

        # filter each wave record by exposition dir  
        mask = filter_angles_between_azimuths(wDir, azimuth_downwave, 
                                              azimuth_upwave, ocean_side) 
        n_waves = mask.sum()
        
        if n_waves == 0:
            break
  
        wDir_mask = wDir[mask]
        wT_mask = wT[mask]
        wHs_mask = wHs[mask]
        
        #Compute wave power
        shelter_wave_power = 1025 * 9.81**2 * wHs_mask**2 * wT_mask / (64 * np.pi)
        #Compute wave power ratio
        shelter_wave_power_ratio[i_point] = shelter_wave_power.sum()/wave_power.sum()
        
        dir_mean, dir_std = wave_mean_dir(wDir_mask, weight = shelter_wave_power) 
        dy = node_spacing * np.sin(np.deg2rad(dir_mean))
        dx = node_spacing * np.cos(np.deg2rad(dir_mean))

        sc = 1
        if ocean_side == 'right':
            sc = -1
            
        x[i_point+1] = x[i_point] + dx * sc
        y[i_point+1] = y[i_point] - dy * sc
        
        mean_wHs_mask[i_point]= wHs_mask.mean() #A
        mean_wT_mask[i_point]= wT_mask.mean()   #A
        mean_wDir_mask[i_point]= dir_mean       #A
        mean_stdDir_mask[i_point]= dir_std       #A
        n_mask[i_point] = n_waves
        
        
    mean_waves_mask = np.vstack([n_mask, mean_wHs_mask, mean_wT_mask, 
                                 mean_wDir_mask, mean_stdDir_mask]) 
                    
    return x, y, shelter_wave_power_ratio, mean_waves_mask


def build_sheltered_coastline(linha_costa, fixed_points, waves, param_lst, 
                              optimize_fixed_points = True, ocean_side = 'right'):

    for index, point in fixed_points.iterrows():
        if point.fp == 'upwave':
            fixed_point_upwave = np.squeeze(np.array(point.geometry.coords.xy))
    
        if point.fp == 'downwave':
            fixed_point_downwave = np.squeeze(np.array(point.geometry.coords.xy))
        
    
    
    linha_costa.create_grid(node_spacing, 1000)
    wDir, wHs, wT = wave_builder(waves, param_lst)

    if optimize_fixed_points:
        res = minimize(rms, fixed_point_upwave, args = (wDir, wT, wHs, fixed_point_downwave, linha_costa, node_spacing, ocean_side), method = 'Nelder-Mead')
        fixed_point_upwave = res.x
    x, y, swpr, mean_waves_mask = shoreline_builder(fixed_point_upwave, wDir, wT, wHs, fixed_point_downwave, linha_costa, node_spacing, ocean_side = ocean_side)
    
     
    rms_value = rms(fixed_point_upwave, wDir, wT, wHs, fixed_point_downwave, 
                    linha_costa, node_spacing, ocean_side) #A

    return x,y, fixed_point_upwave, swpr, mean_waves_mask, rms_value

#%%


"""
INPUT OPTIONS
"""

optimize_fixed_points = True

option_index = 0 # index of the option file listed in the option_list

# the options pkl file are created with create_input_dict script
# file includes location of wave and coastline files and options
option_list = ('optionsPT.pkl', 'optionsNarrabeen.pkl', 
           'optionsCooloola.pkl', 'optionsFlori.pkl', 
           'optionsPontaNegra.pkl', 'optionsBoggoms.pkl',
           'optionsGamtoos.pkl', 'optionsPismo.pkl', 
           'optionsFamJim.pkl', 'optionsMarron.pkl', 
           'optionsPuertoHuarmey.pkl', 'optionsAgraria.pkl',
           'optionsParaiso.pkl', 
           'optionsSensivityMatrix.pkl','optionsSensivityMatrixReta.pkl',
            'optionsPT_conc.pkl','optionsCono1.pkl', 'optionsVictoriaBay.pkl',)
option_list_dir = 'support/'

short_name = str(option_list[option_index][7:-4])
print ('Processing coastline : ' + short_name + '\n')

readfile = open((option_list_dir + option_list[option_index]), 'rb')
options = pickle.load(readfile)
fixed_points =  gpd.read_file(options['fixed_points_file'])
linha_costa = SEMGrid.SEMLine(filename = options['coastline_file'], 
              x_plot = True, xc_plot = False, shp_plot = True, 
              reverse = options['reverse'], x_annotation = False, shp_marker = None)

waves = wts.WaveTimeSeries(filename = options['wave_file'],
              datafile_type = 'era5', lat = options['lat'], long = options['long'], label_style = 'default')



# # Sensitive analysis to shorter wavetimeseries 
# # simulate with  waves between 2015 and 2022 
# from datetime import datetime
# waves_temp = wts.WaveTimeSeries(filename = options['wave_file'],
#               datafile_type = 'era5', lat = options['lat'], long = options['long'], label_style = 'default')
# # datestart =  datetime(1979, 1, 1, 0, 0, 0)
# datestart =  datetime(2015, 1, 1, 0, 0, 0)
# dateend =  datetime(2021, 12, 31, 21, 0, 0)
# waves = waves_temp.cut(datestart, dateend)




replace_sea = False
if replace_sea:
    sea = wts.WaveTimeSeries(datafile_type = 'synthetic', 
        mean_significant_wave_height = 0.01, mean_wave_period = 4, 
        mean_wave_direction = 30, wave_direction_distribution = 'normal',
        wave_height_distribution = 'constant',
        standard_deviation_wave_direction = 100, 
        number_of_waves = waves.wave_data.shape[0])
    waves.wave_data.Hs_ww = sea.wave_data.Hs.values
    waves.wave_data.Dir_ww = sea.wave_data.Dir.values
    waves.wave_data.Tm_ww = sea.wave_data.Tp.values
  
desc = waves.describe()
param_lst = options['wave_parameter_lst']
node_spacing = options['node_spacing']


for index, point in fixed_points.iterrows():
    if point.fp == 'downwave':
        fixed_point_downwave = np.squeeze(np.array(point.geometry.coords.xy))
    elif point.fp == 'upwave':
        fixed_point_upwaveO = np.squeeze(np.array(point.geometry.coords.xy))  #A 

linha_costa.create_grid(node_spacing, 1000)
fig, ax = plt.subplots()
linha_costa.plot()

x, y, fixed_point_upwave, swpr, waves_mask, rms_value = build_sheltered_coastline(linha_costa,  fixed_points, 
                                                            waves, param_lst,  optimize_fixed_points = optimize_fixed_points, ocean_side = options['ocean_side'])

upwave_displac = geometry.Point(fixed_point_upwaveO).distance(geometry.Point(fixed_point_upwave)) #A

ax.plot(x, y, 'r')

 
ax.plot(fixed_point_upwave[0], fixed_point_upwave[1], '+r')
ax.plot(fixed_point_downwave[0], fixed_point_downwave[1], '+g')



#%% Save shapefiles with the results  
# Filenames


if len(param_lst) == 1 and len(param_lst[0])==0:
    wave_code = 'Total'
elif len(param_lst) == 1 and len(param_lst[0])>0:
    wave_code = 'sw'
elif len(param_lst) == 2:
    wave_code = 'Sea_swell'
elif len(param_lst) == 4:
    wave_code ='sw3p'
    
print ('\n Processing coastline :      '+ short_name + 
        '\n Wave conditions :           '+ wave_code + 
        '\n Optimize fixed points :     '+ str(optimize_fixed_points) +
        # '\n Average angular dispersion (deg) :  '+ str(round(np.nanmean(waves_mask[4,:-1]), 2))+
        '\n Upwave fixed point displacement :  ' + str(round(upwave_displac))+
        '\n Coastline position rms error (m) :  '+ str(round(rms_value, 2)))


seg_lines =[]
crs = linha_costa.crs


savingDir='data/BPM_results/' 
# savingDir='data/BPM_results_SensYrs/' 
# savingDir='data/BPM_results_SensTP/' 


if optimize_fixed_points:
    # New coastline
    lineFilename = (savingDir + short_name + '_' + wave_code + 'Op.shp') #analysis_NM
  
    # New upwave fixed point position 
    upwave_pointData = pd.DataFrame({'geometry': [geometry.Point(fixed_point_upwave)],
                                      'upW_displc': round(upwave_displac)})
    pointFilename = (savingDir + short_name + '_' + wave_code + 'Op_pt.shp') 
    newUPc = gpd.GeoDataFrame(crs = crs, geometry = upwave_pointData['geometry'], data = upwave_pointData)
    newUPc.to_file(driver = 'ESRI Shapefile', filename = (pointFilename))
else:
    lineFilename = (savingDir + short_name + '_' + wave_code + '.shp')
    

for a, b, c, d, in zip(x[:-1], y[:-1], x[1:], y[1:]):
      seg_lines.append(geometry.LineString([geometry.Point([a, b]), geometry.Point([c, d])]))
 
seg_linesData = pd.DataFrame({'geometry': seg_lines, 'swpr': swpr[:-1],'n_waves': waves_mask[0,:-1], 'mean_wHs': waves_mask[1,:-1], 
                              'mean_wT': waves_mask[2,:-1], 'mean_wDir': waves_mask[3,:-1], 
                              'rms_value': round(rms_value, 2),'upW_displc': round(upwave_displac), 'std_wDir': waves_mask[4,:-1]})
newCl = gpd.GeoDataFrame(crs = crs, geometry = seg_linesData['geometry'], data = seg_linesData) 
newCl.to_file(driver = 'ESRI Shapefile', filename = lineFilename )
   


#%% 
waves.plot_windrose()
desc = waves.describe()

