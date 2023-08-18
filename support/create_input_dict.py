# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 16:28:52 2022

@author: amasilva
"""

import pickle

saveDir = ''

#%% Portugal
optionsPT = {
    'fixed_points_file': 'data/in_data/geomorph/PT_TroiaSines_FP_tm.shp',
    'coastline_file': 'data/in_data/geomorph/PT_TroiaSines_CL_tm.shp',
   'wave_file': 'data/in_data/waves/PT_1979_2021_sea_and_swell.nc',
  # 'wave_file': 'data/in_data/waves/UK_1979_2021_sea_and_swell.nc',
  'lat': 38.5,
  'long': -10,
    'wave_parameter_lst':[''],
    # 'wave_parameter_lst':['_sw'],
  # 'wave_parameter_lst':['_sw', '_ww'],
    # 'wave_parameter_lst': ['_sw_p1','_sw_p2', '_sw_p3', '_ww'],
  'ocean_side': 'left',
  'reverse': False,
  'node_spacing' : 100
  
}

optFile = open((saveDir + 'optionsPT.pkl'), 'wb')
pickle.dump(optionsPT, optFile)
optFile.close()

#%% Portugal offshore points sensibility analysis
# optionsPT = {
#    'fixed_points_file': 'data/in_data/geomorph/PT_TroiaSines_FP_utm.shp',
#     'coastline_file': 'data/in_data/geomorph/PT_TroiaSines_CL_utm.shp',
#    'wave_file': 'data/in_data/waves/PT_1979_2021_sea_and_swell.nc',
#   # 'wave_file': 'data/in_data/waves/UK_1979_2021_sea_and_swell.nc',
#   'lat': 38.5,
#   'long': -10,
#     # 'wave_parameter_lst':[''],
#     # 'wave_parameter_lst':['_sw'],
#   'wave_parameter_lst':['_sw', '_ww'],
#     # 'wave_parameter_lst': ['_sw_p1','_sw_p2', '_sw_p3', '_ww'],
#   'ocean_side': 'left',
#   'reverse': False,
#   'node_spacing' : 100
  
# }

# optFile = open((saveDir + 'optionsPT_offshore_pt.pkl'), 'wb')
# pickle.dump(optionsPT, optFile)
# optFile.close()



#%% Australia
optionsNarrabeen = {
  'fixed_points_file': 'data/in_data/geomorph/Aus_Narrabeen_FP_utm.shp',
  'coastline_file': 'data/in_data/geomorph/Aus_Narrabeen_CL_utm.shp',
  'wave_file': 'data/in_data/waves/Aust_1979_2021_sea_and_swell.nc',
  'lat':  -33.5,
  'long': 151.5,
    # 'wave_parameter_lst':[''],
    # 'wave_parameter_lst':['_sw'],
    'wave_parameter_lst':['_sw', '_ww'],
    # 'wave_parameter_lst': ['_sw_p1','_sw_p2', '_sw_p3', '_ww'],
   'ocean_side': 'left',
  'reverse': False,
  'node_spacing' : 10
  
}

optFile = open((saveDir + 'optionsNarrabeen.pkl'), 'wb')
pickle.dump(optionsNarrabeen, optFile)
optFile.close()


optionsCooloola = {
  'fixed_points_file': 'data/in_data/geomorph/Aus_Cooloola_FP_utm.shp',
  'coastline_file': 'data/in_data/geomorph/Aus_Cooloola_CL_utm.shp',
  'wave_file': 'data/in_data/waves/Aust_1979_2021_sea_and_swell.nc',
  'lat':  -26,
  'long': 153.5,
    # 'wave_parameter_lst':[''],
    # 'wave_parameter_lst':['_sw'],
    'wave_parameter_lst':['_sw', '_ww'],
    # 'wave_parameter_lst': ['_sw_p1','_sw_p2', '_sw_p3', '_ww'],
   'ocean_side': 'left',
  'reverse': True,
  'node_spacing' : 100
  
}

optFile = open((saveDir + 'optionsCooloola.pkl'), 'wb')
pickle.dump(optionsCooloola, optFile)
optFile.close()

#%% Brasil
optionsFlori = {
  'fixed_points_file': 'data/in_data/geomorph/Br_Florianopolis_FP_utm.shp',
  'coastline_file': 'data/in_data/geomorph/Br_Florianopolis_CL_utm.shp',
  'wave_file': 'data/in_data/waves/Br_1979_2021_sea_and_swell.nc',
  'lat':  -27.5,
  'long': -48,
    'wave_parameter_lst':[''],
    # 'wave_parameter_lst':['_sw'],
    # 'wave_parameter_lst':['_sw', '_ww'],
    # 'wave_parameter_lst': ['_sw_p1','_sw_p2', '_sw_p3', '_ww'],
  'ocean_side': 'left',
  'reverse': False,
  'node_spacing' : 20
  
}

optFile = open((saveDir + 'optionsFlori.pkl'), 'wb')
pickle.dump(optionsFlori, optFile)
optFile.close()


optionsPontaNegra = {
  'fixed_points_file': 'data/in_data/geomorph/Br_PontaNegra_FP_utm.shp',
  'coastline_file': 'data/in_data/geomorph/Br_PontaNegra_CL_utm.shp',
  'wave_file': 'data/in_data/waves/BrPN_1979_2021_sea_and_swell.nc',
  'lat':  -6,
  'long': -34.5,
    'wave_parameter_lst':[''],
    # 'wave_parameter_lst':['_sw'],
    # 'wave_parameter_lst':['_sw', '_ww'],
    # 'wave_parameter_lst': ['_sw_p1','_sw_p2', '_sw_p3', '_ww'],
  'ocean_side': 'left',
  'reverse': True,
  'node_spacing' : 10
  
}

optFile = open((saveDir + 'optionsPontaNegra.pkl'), 'wb')
pickle.dump(optionsPontaNegra, optFile)
optFile.close()
#%% Africa
optionsBoggoms = {
  'fixed_points_file': 'data/in_data/geomorph/AfrS_Boggoms_FP_utm.shp',
  'coastline_file': 'data/in_data/geomorph/AfrS_Boggoms_CL_utm.shp',
  'wave_file': 'data/in_data/waves/AfrS_1979_2021_sea_and_swell.nc',
  'lat':  -34.5,
  'long': 22,
    'wave_parameter_lst':[''],
    # 'wave_parameter_lst':['_sw'],
    # 'wave_parameter_lst':['_sw', '_ww'],
    # 'wave_parameter_lst': ['_sw_p1','_sw_p2', '_sw_p3', '_ww'],
  'ocean_side': 'left',
  'reverse': True,
  'node_spacing' : 50
}

optFile = open((saveDir + 'optionsBoggoms.pkl'), 'wb')
pickle.dump(optionsBoggoms, optFile)
optFile.close()


# optionsVictoriaBay = {
#   'fixed_points_file': 'data/shapefiles/AfrS_VictoriaBay_FP_utm.shp',
#   'coastline_file': 'data/shapefiles/AfrS_VictoriaBay_CL2_utm.shp',
#   'wave_file': 'data/in_data/waves/AfrS_1979_2021_sea_and_swell.nc',
#   'lat':  -34.5,
#   'long': 22.5,
#     # 'wave_parameter_lst':[''],
#     # 'wave_parameter_lst':['_sw'],
#     'wave_parameter_lst':['_sw', '_ww'],
#     # 'wave_parameter_lst': ['_sw_p1','_sw_p2', '_sw_p3', '_ww'],
#   'ocean_side': 'left',
#   'reverse': True,
#   'node_spacing' : 50
# }

# optFile = open((saveDir + 'optionsVictoriaBay.pkl'), 'wb')
# pickle.dump(optionsVictoriaBay, optFile)
# optFile.close()



optionsGamtoos = {
   'fixed_points_file': 'data/in_data/geomorph/AfrS_Gamtoos_FP_utm.shp',
   'coastline_file': 'data/in_data/geomorph/AfrS_Gamtoos_CL_utm.shp',
  'wave_file': 'data/in_data/waves/AfrS_1979_2021_sea_and_swell.nc',
  'lat':  -34.5,
   'long': 25.5,
    'wave_parameter_lst':[''],
    # 'wave_parameter_lst':['_sw'],
    # 'wave_parameter_lst':['_sw', '_ww'],
    # 'wave_parameter_lst': ['_sw_p1','_sw_p2', '_sw_p3', '_ww'],
   'ocean_side': 'left',
  'reverse': True,
  'node_spacing' : 100
  
}

optFile = open((saveDir + 'optionsGamtoos.pkl'), 'wb')
pickle.dump(optionsGamtoos, optFile)
optFile.close()




#%% Califórnia e Baixa Califórnia
optionsPismo = {
    'fixed_points_file': 'data/in_data/geomorph/Calif_Pismo_FP_utm.shp',
   'coastline_file': 'data/in_data/geomorph/Calif_Pismo_CL_utm.shp',
   'wave_file': 'data/waves/AmerN_1979_2021_sea_and_swell.nc',
  'lat':  35,
  'long': -121.5,
    'wave_parameter_lst':[''],
    # 'wave_parameter_lst':['_sw'],
    # 'wave_parameter_lst':['_sw', '_ww'],
    # 'wave_parameter_lst': ['_sw_p1','_sw_p2', '_sw_p3', '_ww'],
  'ocean_side': 'left',
  'reverse': True,
  'node_spacing' : 50
  
}

optFile = open((saveDir + 'optionsPismo.pkl'), 'wb')
pickle.dump(optionsPismo, optFile)
optFile.close()


optionsFamJim = {
    'fixed_points_file': 'data/in_data/geomorph/BaixCalif_FamJim_FP_utm.shp',
    'coastline_file': 'data/in_data/geomorph/BaixCalif_FamJim_CL_utm.shp',
  'wave_file': 'data/in_data/waves/AmerN_1979_2021_sea_and_swell.nc',
  'lat':  30.5,
   'long': -116.5,
    'wave_parameter_lst':[''],
   # 'wave_parameter_lst':['_sw'],
    # 'wave_parameter_lst':['_sw', '_ww'],
    # 'wave_parameter_lst': ['_sw_p1','_sw_p2', '_sw_p3', '_ww'],
 'ocean_side': 'left',
  'reverse': False,
  'node_spacing' : 10
  
}

optFile = open((saveDir + 'optionsFamJim.pkl'), 'wb')
pickle.dump(optionsFamJim, optFile)
optFile.close()


optionsMarron = {
  'fixed_points_file': 'data/in_data/geomorph/BaixCalif_Marron_FP_utm.shp',
  'coastline_file': 'data/in_data/geomorph/BaixCalif_Marron_CL_utm.shp',
  'wave_file': 'data/in_data/waves/AmerN_1979_2021_sea_and_swell.nc',
  'lat':  29,
  'long': -115,
    'wave_parameter_lst':[''],
    # 'wave_parameter_lst':['_sw'],
    # 'wave_parameter_lst':['_sw', '_ww'],
    # 'wave_parameter_lst': ['_sw_p1','_sw_p2', '_sw_p3', '_ww'],
  'ocean_side': 'left',
  'reverse': False,
  'node_spacing' : 10
  
}

optFile = open((saveDir + 'optionsMarron.pkl'), 'wb')
pickle.dump(optionsMarron, optFile)
optFile.close()


#%% América Sul - Pacífico
optionsPuertoHuarmey = {
  'fixed_points_file': 'data/in_data/geomorph/Pacif_PuertoHuarmey_FP_utm.shp',
  'coastline_file': 'data/in_data/geomorph/Pacif_PuertoHuarmey_CL_utm.shp',
   'wave_file': 'data/in_data/waves/AmerS_Pacif_1979_2021_sea_and_swell.nc',
  'lat':  -10,
  'long': -78.5,
    'wave_parameter_lst':[''],
    # 'wave_parameter_lst':['_sw'],
    # 'wave_parameter_lst':['_sw', '_ww'],
    # 'wave_parameter_lst': ['_sw_p1','_sw_p2', '_sw_p3', '_ww'],
   'ocean_side': 'right',
  'reverse': True,
  'node_spacing' : 10
}

optFile = open((saveDir + 'optionsPuertoHuarmey.pkl'), 'wb')
pickle.dump(optionsPuertoHuarmey, optFile)
optFile.close()


optionsAgraria = {
   'fixed_points_file': 'data/in_data/geomorph/Pacif_Agraria_FP_utm.shp',
   'coastline_file': 'data/in_datadata/waves/Pacif_Agraria_CL_utm.shp',
  'wave_file': 'data/in_data/waves/AmerS_Pacif_1979_2021_sea_and_swell.nc',
  'lat':  -11,
  'long': -78,
    # 'wave_parameter_lst':[''],
    # 'wave_parameter_lst':['_sw'],
    'wave_parameter_lst':['_sw', '_ww'],
    # 'wave_parameter_lst': ['_sw_p1','_sw_p2', '_sw_p3', '_ww'],
   'ocean_side': 'right',
  'reverse': True,
  'node_spacing' : 10
}

optFile = open((saveDir + 'optionsAgraria.pkl'), 'wb')
pickle.dump(optionsAgraria, optFile)
optFile.close()

optionsParaiso = {
  'fixed_points_file': 'data/in_data/geomorph/Pacif_Paraiso_FP_utm.shp',
  'coastline_file': 'data/in_data/geomorph/Pacif_Paraiso_CL_utm.shp',
   'wave_file': 'data/in_data/waves/AmerS_Pacif_1979_2021_sea_and_swell.nc',
  'lat':  -11,
  'long': -78,
    # 'wave_parameter_lst':[''],
    # 'wave_parameter_lst':['_sw'],
    'wave_parameter_lst':['_sw', '_ww'],
    # 'wave_parameter_lst': ['_sw_p1','_sw_p2', '_sw_p3', '_ww'],
 'ocean_side': 'right',
  'reverse': True, 
  'node_spacing' : 10
}

optFile = open((saveDir + 'optionsParaiso.pkl'), 'wb')
pickle.dump(optionsParaiso, optFile)
optFile.close()

#%% PT_Conceptual
# optionsPT_conc = {
#   'fixed_points_file': 'data/shapefiles_NM/PT_Conceptual_FP.shp',
#    'coastline_file': 'data/shapefiles_NM/PT_Conceptual.shp',
#   'wave_file': 'data/in_data/waves/PT_1979_2021_sea_and_swell.nc',
#   'lat':  38,
#   'long': -10,
#     'wave_parameter_lst':[''],
#     # 'wave_parameter_lst':['_sw'],
#     # 'wave_parameter_lst':['_sw', '_ww'],
#     # 'wave_parameter_lst': ['_sw_p1','_sw_p2', '_sw_p3', '_ww'],
#   'ocean_side': 'left',
#   'reverse': False
  
# }

# optFile = open((saveDir + 'optionsPT_conc.pkl'), 'wb')
# pickle.dump(optionsPT_conc, optFile)
# optFile.close()



#%% Concetual sensitivity Matrix
# optionsSensivityMatrix = {
#   'fixed_points_file': 'data/shapefiles/concetual_FP.shp',
#   'coastline_file': 'data/shapefiles/concetual_CL.shp',
#   'wave_file': 'data/in_data/waves/PT_1979_2021_sea_and_swell.nc',
#   'lat':  37,
#   'long': -8.5,
#   'wave_parameter_lst': [''],
#   'ocean_side': 'right',
#   'reverse': False
    
# }

# optFile = open((saveDir + 'optionsSensivityMatrix.pkl'), 'wb')
# pickle.dump(optionsSensivityMatrix, optFile)
# optFile.close()

#%% Concetual sensitivity Matrix Reta
# optionsSensivityMatrix = {
#   'fixed_points_file': 'data/shapefiles/concetual_FP.shp',
#   'coastline_file': 'data/shapefiles/concetual_CL_reta.shp',
#   'wave_file': 'data/in_data/waves/PT_1979_2021_sea_and_swell.nc',
#   'lat':  37,
#   'long': -8.5,
#   'wave_parameter_lst': [''],
#   'ocean_side': 'right',
#   'reverse': False
  
  
# }

# optFile = open((saveDir + 'optionsSensivityMatrixReta.pkl'), 'wb')
# pickle.dump(optionsSensivityMatrix, optFile)
# optFile.close()


