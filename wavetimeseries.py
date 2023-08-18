# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 22:01:46 2020

@author: DISEPLA - FCUL rui / cristina / ana
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker    
from scipy.constants import g, pi
import xarray as xr
import copy
import wavepy as wv
import scipy.io
import sys

from numpy.random import default_rng
    
class WaveTimeSeries:
    def __init__(self, **kwargs):
        self.filename = []

        self.datafile_type = 'excel'
        self.label_style = 'default'
        self.show_label_units = True
        self.missing_values = []
        self.show_legend = False
        self.__init_wave_labels()   
        self.time_fix = True
        
        #polar plot definitions
        self.n_dir_bins = 16
        self.rose_colormap = cm.jet
        
        #frequency table
        self.relative_freq = True
        
        self.breaker_index = 0.78
        self.breaking = False
        
        self.rho = 1025
        
        self.lat = 39.
        self.long = -9.5
        
        self.dpi_figures = 300
        
        self.drop_expver = True
        
        
        #synthetic wave parameters

        # wave direction
        self.wave_direction_distribution = 'von Mises'
        self.mean_wave_direction = 30
        self.k_parameter_wave_direction = 20
        self.standard_deviation_wave_direction = 20
  
        # wave height
        self.wave_height_distribution = 'log normal'
        self.mean_significant_wave_height = 1;
        self.standard_deviation_significant_wave_height = 0.1
        
        # wave period
        self.wave_period_distribution = 'normal'
        self.mean_wave_period = 6
        self.standard_deviation_wave_period = 0.5
        
        self.number_of_waves = 10000
        
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.datafile_type == 'excel':
           data = pd.read_excel(self.filename)
           for missing_value in self.missing_values:
               data[data == missing_value] = np.nan
               data[data['MWD']== 999] = np.nan #A
               data = data.dropna(axis = 0)
               
           self.wave_data = data
           self.wave_data = self.wave_data.set_index('time')
           
        elif self.datafile_type == 'synthetic':
            
            rng = default_rng()

            
            # wave direction
            if self.wave_direction_distribution == 'von Mises':
               wave_dir  = np.degrees(np.random.vonmises(np.radians(self.mean_wave_direction), self.k_parameter_wave_direction, self.number_of_waves))
               wave_dir[wave_dir <0] += 360 # always return positve angles
            elif self.wave_direction_distribution == 'constant':
               wave_dir  = np.ones(self.number_of_waves) * self.mean_wave_direction
            elif self.wave_direction_distribution == 'normal':
               wave_dir  = np.random.normal(self.mean_wave_direction, self.standard_deviation_wave_direction, self.number_of_waves)
               wave_dir[wave_dir <0] += 360 # always return positve angles
            else:
                sys.exit("Wave direction - ", self.wave_direction_distribution, " - distribution not implemented")
                      
            # wave height
            if self.wave_height_distribution == 'log normal':
                # see https://blogs.sas.com/content/iml/2014/06/04/simulate-lognormal-data-with-specified-mean-and-variance.html
               phi = np.sqrt(self.standard_deviation_significant_wave_height**2 + self.mean_significant_wave_height**2)
               mean = np.log(self.mean_significant_wave_height**2 /phi);
               sd = np.sqrt(np.log(phi**2 / self.mean_significant_wave_height**2))
               wave_height = rng.lognormal(mean, sd, self.number_of_waves)
            elif self.wave_height_distribution == 'constant':
               wave_height  = np.ones(self.number_of_waves) * self.mean_significant_wave_height
            else:
                sys.exit("Wave height distribution not implemented")
            
            # wave period
            if self.wave_period_distribution == 'normal':
               wave_period = np.random.normal(self.mean_wave_period, self.standard_deviation_wave_period, self.number_of_waves)
            elif self.wave_period_distribution == 'constant':
               wave_period  = np.ones(self.number_of_waves) * self.mean_wave_period
            else:
                sys.exit("Wave period distribution not implemented")
              
               
            self.wave_data = pd.DataFrame({'Dir': wave_dir, 'Hs': wave_height, 'Tm': wave_period, 'Tp': wave_period})
            
        elif self.datafile_type == 'hdf':
           self.wave_data = pd.read_hdf('output.hdf', 'wave_data')
        elif self.datafile_type == 'era5':
            #https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form
            ds = xr.open_dataset(self.filename)
            ds_sel = ds.sel(longitude = self.long, latitude = self.lat)
            # mwd - mean direction
            # pp1d - peak wave period
            # swh - significant height of combined wind waves and swell
            # wsp - Wave spectral peakedness 
            wave_data = ds_sel.to_dataframe()
            #if self.drop_expver:
            # wave_data = wave_data.droplevel('expver')
            wave_data.dropna(inplace = True)
            
            self.wave_data = wave_data.drop(['latitude', 'longitude'], axis = 1)
            
        elif self.datafile_type == 'netcdf_copernicus':
            # buoy data from EMODnet
            # http://www.emodnet-physics.eu/map/
            # PRODUCT USER MANUAL
            # https://archimer.ifremer.fr/doc/00437/54853/56332.pdf
            xr_wave_data = xr.open_dataset(self.filename)
            
            #'VHM0 -  Spectral significant wave height (Hm0)
            #'VTM02 - Spectral moments (0,2) wave period (Tm02)
            #'VMDR  - Mean wave direction from (Mdir)'
            #'VTPK  - Wave period at spectral peak / peak period (Tp)
            wave_data = xr_wave_data[['VHM0', 'VTM02', 'VMDR', 'VTPK']].to_dataframe()
            xr_wave_data.close()
            self.wave_data = wave_data.xs(1, level='DEPTH')
        
        else:
            sys.exit("Wave data file type not implemented")
                                  
        
    #normalize wave parameters names
        colum_names = map(str.lower, list(self.wave_data.columns))
        colum_names = ['Tm' if x == 'tm' or x=='tz' or x =='tmed' or x == 'mwp'
                       or x == 'vtm02' or x == 'apd' else x for x in colum_names]
        colum_names = ['Tp' if x == 'pp1d' or x =='tpeak' or x =='tp' or 
                       x == 'vtpk' or x == 'peak wave period' or x == 'dpd'
                       else x for x in colum_names]
        colum_names = ['Tm_sw' if x =='mpts' else x for x in colum_names]
        colum_names = ['Tm_ww' if x =='mpww' else x for x in colum_names]
        colum_names = ['Hs' if x =='hsig' or x =='swh' or  x == 'hm0' 
                       or x == 'hs' or x == 'vhm0'or x=='significant wave height' 
                       or x == 'wvht' else x for x in colum_names]
        colum_names = ['Hs_sw' if x =='shts' else x for x in colum_names]
        colum_names = ['Hs_ww' if x =='shww' else x for x in colum_names]
        colum_names = ['Dir' if x == 'dm' or x=='mwd' or x =='dir' or x == 'DirMed' 
                       or x == 'vmdr' or x=='weighted mean wave direction' 
                       else x for x in colum_names]
        colum_names = ['Dir_sw' if x == 'mdts' else x for x in colum_names]
        colum_names = ['Dir_ww' if x == 'mdww' else x for x in colum_names]
        colum_names = ['DSp' if x == 'wdw' or x == 'weighted mean spreading width' 
                       else x for x in colum_names]
        colum_names = ['DSp_sw' if x == 'dwps' else x for x in colum_names]
        colum_names = ['DSp_ww' if x == 'dwww' else x for x in colum_names]
        colum_names = ['Hs_sw_p1' if x == 'p140121' else x for x in colum_names]
        colum_names = ['Dir_sw_p1' if x == 'p140122' else x for x in colum_names]
        colum_names = ['Tm_sw_p1' if x == 'p140123' else x for x in colum_names]
        colum_names = ['Hs_sw_p2' if x == 'p140124' else x for x in colum_names]
        colum_names = ['Dir_sw_p2' if x == 'p140125' else x for x in colum_names]
        colum_names = ['Tm_sw_p2' if x == 'p140126' else x for x in colum_names]
        colum_names = ['Hs_sw_p3' if x == 'p140127' else x for x in colum_names]
        colum_names = ['Dir_sw_p3' if x == 'p140128' else x for x in colum_names]
        colum_names = ['Tm_sw_p3' if x == 'p140129' else x for x in colum_names]
        
      
        
        self.wave_data.columns = colum_names
       
        
        
    def __init_wave_labels(self):
        wave_param_names = ['Hs', 'Hrms', 'Hs_sw', 'Hs_ww', 'Tp', 'Tm', 'Tm_sw', 'Tm_ww', 'Dir', 'Dir_sw', 'Dir_ww', 'DSp', 'DSp_sw', 'DSp_ww']
        wave_param_data = np.array([['Hs', 'Hrms', 'Hs_sw', 'Hs_ww', 'Tp', 'Tm', 'Tm_sw', 'Tm_ww', r'$\ \theta$', r'$\ \theta$', r'$\ \theta$','DSp', 'DSp_sw', 'DSp_ww'],
                            ['swh', 'hrms', 'shts', 'shww', 'pwp', 'pp1d', 'mwd', 'mpts', 'mpww', 'mdts', 'mdww', 'wdw', 'dwps', 'dwww'],
                            ['Hm0', 'Hrms', 'Hs_sw', 'Hs_ww', 'Tp', 'T0', 'Tm_sw', 'Tm_ww', r'$\ \theta$', r'$\ \theta$', r'$\ \theta$', 'wdw', 'dwps', 'dwww'],
                            ['significant wave height', 'root-mean-square wave height', 'significant heigth total swell', 'significant heigth wind waves', 'peak wave period', 'mean wave period', 'mean period total swell' , 'mean period wind waves' , 'mean wave direction', 'mean direction total swell', 'mean direction wind waves', 'wave spectral directional width', 'wave spectral directional width total swell', 'wave spectral directional width wind waves'],
                            ['altura significativa', 'altura média quadrática', 'altura significativa ondulação', 'altura significativa vaga', 'período de pico', 'período médio', 'período médio ondulação', 'período médio vaga' , 'direção média', 'direção média ondulação', 'direção média vaga', 'dispersão direcional das ondas', 'dispersão direcional ondulação', 'dispersão direcional vaga']])
        units = np.array([['m', 'm', 'm', 'm', 's', 's', 's', 's' , u'\u00b0', u'\u00b0', u'\u00b0', '1', '1', '1']])
        label_type = ['default', 'ERA5', 'spc', 'en' , 'pt', 'units' ]
        

        self.wave_labels = pd.DataFrame(data =  np.concatenate([wave_param_data, units]), columns = wave_param_names, index = label_type)
        
               
    def dt_f(self, unit = 'h'): #return a scalar if dt is constant
        dt = np.diff(self.wave_data.index)/np.timedelta64(1, unit)
        dt_max = np.max(dt)
        dt_min = np.min(dt)
        if dt_max == dt_min:
            return dt_max
        else:
            return dt
    
    def describe(self):
        stats = self.wave_data.describe()
        m_circular, std_circular = self.dir_mean()
        stats['Dir'].values[1] = m_circular
        stats['Dir'].values[2] = std_circular
        
        stats['Dir_WP'] = np.nan
        m_circular, std_circular = self.dir_mean(weight_mode = 'wave_power')
        stats['Dir_WP'].values[0] = stats['Dir'].values[0]
        stats['Dir_WP'].values[1] = m_circular
        stats['Dir_WP'].values[2] = std_circular
        return stats
    
    def dir_mean(self, parameter = 'Dir', weight_mode = 'constant'):
        dir = self.wave_data[parameter]
        if weight_mode == 'wave_power':
            weight = self.wave_power_offshore(t_parameter = 'Tp') 
        else:
            weight = np.ones(len(dir))
        
        x = np.sum(weight * np.cos(np.radians(dir)))
        y = np.sum(weight * np.sin(np.radians(dir)))
        dir_m = np.degrees(np.arctan2(y, x))
        if dir_m < 0: 
            dir_m += 360
        std_m = np.degrees(np.sqrt(-2 * np.log(np.sqrt(x**2 + y**2) / weight.sum())))
        return dir_m, std_m
        
    def dir_mean_wave_power(self, parameter = 'Dir', t_parameter = 'Tp'):
        return self.dir_mean(parameter, weight = 'wave_power')

    
    def maxima(self, freq = 'Y', parameter = 'Hs', plot = False):
        hs_max = self.wave_data[parameter].groupby(pd.Grouper(freq = freq)).max()
        hs_max.index = self.wave_data[parameter].groupby(pd.Grouper(freq = freq)).idxmax()
        return hs_max
        
    def plot_timeseries(self, parameter = 'Hs', ax = False):
        if not ax:
            fig, ax = plt.subplots()
        if not(self.wave_data.empty):
            ax = self.wave_data.plot(y = parameter, legend = self.show_legend, ax = ax)
            ax.set_xlabel('')
            self.axis_labels(ax, y = parameter)
        return ax

    def plot_all_timeseries(self, t_parameter = 'Tp', ax = False):
        if not np.any(ax):
            fig, ax = plt.subplots(nrows = 3, ncols = 1)
        self.plot_timeseries(parameter = 'Hs', ax = ax[0])
        
        if t_parameter == 'Tm':
            self.plot_timeseries(parameter = 'Tm', ax = ax[1])
        else:
            self.plot_timeseries(parameter = 'Tp', ax = ax[1])
            
        self.plot_timeseries(parameter = 'Dir', ax = ax[2])
        plt.tight_layout()
        return ax
        
    def axis_labels(self, ax, x = False, y = False):
        if x:
            units = self.show_label_units * (' ('+ self.wave_labels.loc['units', x]+')')
            ax.set_xlabel(self.wave_labels.loc[self.label_style, x] + units)
        if y:
            units = self.show_label_units * (' ('+ self.wave_labels.loc['units', y]+')')
            ax.set_ylabel(self.wave_labels.loc[self.label_style, y] + units)
      
    
    
    
    def joint_distribution(self, type = 'scatter', x = 'Tm', y = 'Hs', plot_steepness_domains = False):
        if type == 'scatter':
            ax = self.wave_data.plot.scatter(x, y,  1)
        elif type == 'histogram':
            ax = plt.subplot(111)
            ax.hist2d(self.wave_data[x], self.wave_data[y], (50,50), cmap = cm.jet)
        
        self.axis_labels(ax, x, y)
        
        if plot_steepness_domains:
            # see Holthuijsen, L. H. (2010). Waves in oceanic and coastal waters. Cambridge university press.
            (x_min, x_max) = ax.get_xlim()
            x_vect = np.linspace(x_min, x_max, 20)
            ax.plot(x_vect,  self.wave_steepness(x_vect, 1/15), color = 'black', linewidth = 0.2)
            ax.plot(x_vect,  self.wave_steepness(x_vect, 1/30), color = 'black', linewidth = 0.2)
            
            ax.text(x_vect[-2], self.wave_steepness(x_vect[-1], 1/15), '1:15', fontsize = 'x-small')
            ax.text(x_vect[-2], self.wave_steepness(x_vect[-1], 1/30), '1:30', fontsize = 'x-small')
            plt.gcf().set_dpi(self.dpi_figures)
        
        return ax
        
    def wave_power_offshore(self, t_parameter = 'Tp'):
        if  t_parameter == 'Tp':
            Te = 0.9 * self.wave_data.Tp # the wave energy period see eqs 41 and 42 in https://www.sciencedirect.com/science/article/pii/S0141118718303821
        elif t_parameter == 'Tm':
            Te = 1.154 * self.wave_data.Tm 
        else:
            sys.exit("Wave period parameter - " + t_parameter + " - not found")
            
        return self.rho * g**2 * self.wave_data.Hs**2 * Te / (64 * pi)
    
    def wave_steepness(self, Tm, steepness):
         return steepness * Tm ** 2 * g / (2 * pi)
     
    def from_edges_to_centers(self, edges):
        return (edges[1:] + edges[:-1]) / 2
        
    def freq_table(self, var1, var2, bin_edges_var1, bin_edges_var2):
        freq, _, _ = np.histogram2d(self.wave_data[var1], self.wave_data[var2], bins=(bin_edges_var1, bin_edges_var2))
        
        if self.relative_freq:
            freq = freq/freq.sum()
            
        bin_centers_var1 = self.from_edges_to_centers(bin_edges_var1)
        bin_centers_var2 = self.from_edges_to_centers(bin_edges_var2)
 
        classes_var1, classes_var2 = np.meshgrid(bin_centers_var2, bin_centers_var1)
        
        freq_in_columns = pd.DataFrame({'freq': freq.flatten(), var2: classes_var1.flatten(), var1: classes_var2.flatten()})
        
        return freq, freq_in_columns
    
    def cut(self, date_stats, date_end, copy_wts = True):
        if copy_wts:
            wts = copy.copy(self)
        else:
            wts = self
        wts.wave_data = wts.wave_data.loc[date_stats:date_end]
        return wts
    
    def freq_windrose(self, bin_edges, dir_parameter = 'Dir', parameter = 'Hs', n_dir_bins = None):
        
        if n_dir_bins == None:
            n_dir_bins = self.n_dir_bins
            
        dir_bin_width = 360/n_dir_bins
              
        #center first bin around direction 0
        dir_bin_edges = np.linspace(dir_bin_width/2, 360-dir_bin_width/2, n_dir_bins)
        dir_bin_edges = np.append(0, dir_bin_edges)
        dir_bin_edges = np.append(dir_bin_edges, 360)
       
        freq, _ = self.freq_table(dir_parameter, parameter, dir_bin_edges, bin_edges)    
         
        #sum the frequency of the first and last bin - right and left off 0
        freq[0, :] = freq[0, :] + freq[-1, :]
        freq = freq[:-1, :]

        #computed center of classes, including the first - around 0
        dir_centers = self.from_edges_to_centers(dir_bin_edges)
        dir_centers = np.append(0, dir_centers[1:-1])
        scalar_centers = self.from_edges_to_centers(bin_edges)
        
        return pd.DataFrame(freq, columns = scalar_centers, index = dir_centers)
    
    def plot_windrose(self, dir_parameter = 'Dir', parameter = 'Hs', colormap = None, n_dir_bins = None, bin_edges = None, 
                      legend = True, show_freq = True, show_axis = True, show_ticks = True, position = False, fig = False, log = False):
        
        if np.sum(bin_edges) == None:
            bin_edges = np.linspace(0, np.ceil(self.wave_data[parameter].max()), 8)
 
        freq_winrose = self.freq_windrose(bin_edges, dir_parameter, parameter, n_dir_bins)
        
        bin_centers = freq_winrose.columns
        dir_centers = freq_winrose.index
        freq = freq_winrose.to_numpy()

        if log: #in case of log scale sum 10 
            
            freq[freq>0] = np.log(freq[freq>0]) + 10 

        freq_cum = freq.cumsum(axis = 1)
        
     
        sector_width_rad = np.radians(np.diff(dir_centers)[0])
        
        if not(position):
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        else:
            if not(fig):
                print('a figure handle is required')
                sys.exit()
            else:
                #[left, bottom, width, height]
                ax = fig.add_axes(position, projection  = 'polar')
                
        
            
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
       
        if colormap == None:
            colormap = self.rose_colormap
        norm = Normalize(vmin = bin_centers[0], vmax = bin_centers[-1])
        colors = colormap(norm(bin_centers))
        
        ax.bar(np.radians(dir_centers), freq[:, 0], bottom=0, width = sector_width_rad, color = colors[0])
        for j in range(1, len(bin_centers)):
            ax.bar(np.radians(dir_centers), freq[:, j], bottom = freq_cum[:,j-1], width = sector_width_rad, color = colors[j])
        
       
        ax.grid(linestyle = ':')
        ax.patch.set_visible(False)    
       
        if not(show_ticks):
            ax.set_xticks([]) 
            
        if not(show_axis):
            ax.grid('off')
            ax.spines['polar'].set_visible(False)
         
        
        
        if show_freq:
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax = 1))
            # ax.tick_params(axis = 'y', which='major', labelsize= 'small', labelcolor='red')
            ax.tick_params(axis = 'y', which='major', labelsize= 'small')
        else:
            ax.set_yticks([]) 
            # ax.tick_params(axis = 'y', which='major', labelsize= 0)
        
        if legend:
            units = self.show_label_units * (' ('+ self.wave_labels.loc['units', parameter]+')')
            handles = [plt.Rectangle((0,0),1,1, color =c ) for c in colors]
            labels = [u"[ {:0.2f} - {:0.2f} [".format(bin_edges[i], bin_edges[i+1]) for i in range(len(bin_centers))]
            ax.legend(handles, labels, fontsize = 'x-small', loc = (1,0), title = self.wave_labels.loc[self.label_style, parameter] + units)
      
        return ax
       
    def to_break(self, dir_bottom = 0, H_stat = 'Hs', h0 = 500):
        wts_breaking = copy.deepcopy(self)
        
        
        wave_break_data = np.zeros((self.wave_data['Hs'].shape[0], 3))
        
        for i, (H, T, Dir) in enumerate(zip(wts_breaking.wave_data['Hs'], wts_breaking.wave_data['Tp'], wts_breaking.wave_data['Dir'])):
            w = wv.Wave(H = H, T = T, dir = Dir, dir_bottom = dir_bottom, H_stat = H_stat, h0 = h0)
            w.break_it()
          
            wave_break_data[i, :] = [w.H, w.T, w.dir]
            


        wts_breaking.wave_data.Hs = wave_break_data[:,0]
        wts_breaking.wave_data.Tp = wave_break_data[:,1]
        wts_breaking.wave_data.Dir = wave_break_data[:,2]
        wts_breaking.breaking = True
        
        return wts_breaking      
    
    def ls_drift(self, k = 0.39, dir_bottom = 0, method = 'CERC'):
        self.rho = 1025
        #sediment properties
        rho_s = 2650
        p = 0.4
        if self.breaking:
            c = self.rho * np.sqrt(g) / (16*np.sqrt(self.breaker_index)*(rho_s - self.rho)*(1-p))
            if method == 'Mil_Homens_et_al':
                L0 = 1.56 * self.wave_data['Tp']**2
                k = 1/(2232*(self.wave_data['Hs']/L0) ** 1.45 + 4.505)
            alpha =  self.wave_data['Dir'] - dir_bottom 
            return k * c * self.wave_data['Hs']**(5/2.) * np.sin(np.radians(2 * alpha)), alpha
      
    def storm_events(self, Hthreshold, duration):
        
        wd = self.wave_data
        wd['label'] = (~wd['Hs'].ge(Hthreshold)).cumsum()
        st = wd[wd['Hs']>= Hthreshold]
        st_count = st.value_counts(st['label'])

        #duration -  duration of storm in h
        self.interval = wd.index.hour[1] - wd.index.hour[0]
        nevents = int(duration / self.interval)
        self.ev = st_count[st_count>= nevents]
        
        self.storms = st.loc[st['label'].isin(self.ev.index)]
        print('storms', self.storms)
        return
    
    def storm_events_num (self):
        print('number of events = ', self.ev.size)
    
    def storm_data_to_excel (self, name):
        self.storms.to_excel(name)
        return
    
    def storm_stats (self):
        stats = self.storms.drop_duplicates('label')
        length = (self.storms.groupby(['label']).size()) * self.interval
        stats = stats.copy()
        stats.index.names = ['start_time']
        stats.loc[:,'duration'] = list(length)
        stats.loc[:, 'Hs_max'] = list(self.storms.groupby(['label']).max(['Hs'])['Hs'])
        stats.loc[:, 'Hs_mean'] = list(self.storms.groupby(['label']).mean(['Hs'])['Hs'])
        
        if self.storms.columns.size == 2:
            print('Data has only Hs variable')
            self.stats = stats.drop(['label'], axis=1)
            print(self.stats)
            
        else:
            stats.loc[:, 'Tp_max'] = list(self.storms.groupby(['label']).max(['Tp'])['Tp'])
            stats.loc[:, 'TP_mean'] = list(self.storms.groupby(['label']).mean(['Tp'])['Tp'])
            stats.loc[:, 'Dir_mean'] = list(self.storms.groupby(['label']).mean(['Dir'])['Dir'])
        
            self.stats = stats.drop(['label'], axis=1)
            print(self.stats)
        return
    
    def storm_stats_to_excel(self, name):
        self.stats.to_excel(name)
        return
    
    

    
    
                                        
                                       