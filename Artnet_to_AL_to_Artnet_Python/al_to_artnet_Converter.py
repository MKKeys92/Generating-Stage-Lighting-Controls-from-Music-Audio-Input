import numpy as np
import colorsys
import json
import argparse
import pyhocon


#######################


def features_72params_to_artnet_DKv03(features_DKv01, artnet_dummy):
    # len = np.shape(features_DKv01)[0]
    # updates every frame, change this for use in matlab replacement for the whole array
    len = 1
    
    AL_to_artnet = np.zeros((len, 8192))
    one_universe_size = 512
    bmfl_group_truss_count = 9
    bmfl_group_sidestage_count = 6
    bmfl_group_floor_count = 7
    jdc1_group_count = 8
    
    # Get Abstraction Layer values for each group
    al_large_len = 18
    al_small_len = 8
    hsv_len = 3

    # Features BMFL #01 for each fixture
    # |:
    # 1 = Dim, 2 = ColorHue, 3 = ColorSat, 4 = Pan, 5 = Tilt,
    # 6 = Shutter, 7 = Zoom, 8 = Iris
    # :|
    #
    # AL Large
    # 01 = Position of Max Intensity
    # 02 = Intensity of Peak absolute
    # 03 = Slope of Peak Intensity
    # 04 = AF Peak Density
    # 05 = AF Peak Simularity
    # 06 = AF Pos 2nd Peak (if exists, otherwise Pos 1st)
    # 07 = AF Intensity of Minima absolute
    # 08 = AF Minima Simularity
    # 09 = Col Hue Lead
    # 10 = Col Sat Lead
    # 11 = Col Hue Back
    # 12 = Col Sat Back
    # 13 = Pos Max Left Pan
    # 14 = Val Max Left Pan
    # 15 = Pos Max Right Pan
    # 16 = Val Max Right Pan
    # 17 = Value of Max Tilt
    # 18 = Value of Min Tilt
    #

    bmfl_al_large_ftruss = np.zeros((1, al_large_len))
    #bmfl_al_large_ftruss = features_DKv01[:, 0:17]
    for i in range(al_large_len):
        bmfl_al_large_ftruss[0,i] = features_DKv01[0,(0+i)]
    
    jdc1_al_small_ftruss = np.zeros((1, al_small_len))
    #jdc1_al_small_ftruss = features_DKv01[:, 18:25]
    for i in range(al_small_len):
        jdc1_al_small_ftruss[0,i] = features_DKv01[0,(18+i)]

    bmfl_al_small_m1truss = np.zeros((1, al_small_len))
    #bmfl_al_small_m1truss = features_DKv01[:, 26:33]
    for i in range(al_small_len):
        bmfl_al_small_m1truss[0,i] = features_DKv01[0,(26+i)]

    jdc1_al_small_m1truss = np.zeros((1, al_small_len))
    #jdc1_al_small_m1truss = features_DKv01[:, 34:41]
    for i in range(al_small_len):
        jdc1_al_small_m1truss[0,i] = features_DKv01[0,(34+i)]
    #print('jdc1_al_small_m1truss')
    #print(np.shape(jdc1_al_small_m1truss))

    bmfl_al_large_btruss = np.zeros((1, al_large_len))
    #bmfl_al_large_btruss = features_DKv01[:, 42:59]
    for i in range(al_large_len):
        bmfl_al_large_btruss[0,i] = features_DKv01[0,(42+i)]
    
    jdc1_al_hsv_btruss = np.zeros((1, hsv_len))
    #jdc1_al_hsv_btruss = features_DKv01[:, 60:62]
    for i in range(hsv_len):
        jdc1_al_hsv_btruss[0,i] = features_DKv01[0,(60+i)]

    jdc1_al_hsv_backstage_floor = np.zeros((1, hsv_len))
    #jdc1_al_hsv_backstage_floor = features_DKv01[:, 63:65]
    for i in range(hsv_len):
        jdc1_al_hsv_backstage_floor[0,i] = features_DKv01[0,(63+i)]

    jdc1_al_hsv_frontstage_floor = np.zeros((1, hsv_len))
    #jdc1_al_hsv_frontstage_floor = features_DKv01[:, 66:68]
    for i in range(hsv_len):
        jdc1_al_hsv_frontstage_floor[0,i] = features_DKv01[0,(66+i)]
    
    bmfl_al_hsv_sidestage_front = np.zeros((1, hsv_len))
    for i in range(hsv_len):
        bmfl_al_hsv_sidestage_front[0,i] = features_DKv01[0,(69+i)]

    control_chan01 = op('control_chan01')[0,0]
    control_chan02 = op('control_chan02')[0,0]
    control_chan03 = op('control_chan03')[0,0]
    control_chan04 = op('control_chan04')[0,0]
    control_chan05 = op('control_chan05')[0,0]
    control_chan06 = op('control_chan06')[0,0]
    control_chan07 = op('control_chan07')[0,0]
    control_chan08 = op('control_chan08')[0,0]
    control_toggle01 = op('control_toggle01')[0,0]
    control_toggle02 = op('control_toggle02')[0,0]
    control_toggle03 = op('control_toggle03')[0,0]
    control_toggle04 = op('control_toggle04')[0,0]
    control_toggle05 = op('control_toggle05')[0,0]
    control_toggle06 = op('control_toggle06')[0,0]
    control_toggle07 = op('control_toggle07')[0,0]
    control_toggle08 = op('control_toggle08')[0,0]

    if control_toggle01 == 0:
        bmfl_al_large_ftruss[0,2] = control_chan01

    # bmfl_al_large_ftruss[0,2] = bmfl_al_large_ftruss[0,2] * control_chan01
    print(bmfl_al_large_ftruss[0,2])

    # Get reduced values for each fixture per group
    bmfl_fixture_specific_ftruss = bmfl_al_to_fixture_specific_large(bmfl_al_large_ftruss, bmfl_group_truss_count)
    bmfl_fixture_specific_m1truss = bmfl_al_to_fixture_specific_small(bmfl_al_small_m1truss, bmfl_group_truss_count)
    bmfl_fixture_specific_btruss = bmfl_al_to_fixture_specific_large(bmfl_al_large_btruss, bmfl_group_truss_count)
    bmfl_fixture_specific_sidestage_front = bmfl_al_to_fixture_specific_hsv(bmfl_al_hsv_sidestage_front, bmfl_group_sidestage_count)

    jdc1_fixture_specific_ftruss = jdc1_al_to_fixture_specific_small(jdc1_al_small_ftruss, jdc1_group_count)
    jdc1_fixture_specific_m1truss = jdc1_al_to_fixture_specific_small(jdc1_al_small_m1truss, jdc1_group_count)
    jdc1_fixture_specific_btruss = jdc1_al_to_fixture_specific_hsv(jdc1_al_hsv_btruss, jdc1_group_count)
    jdc1_fixture_specific_backstage_floor = jdc1_al_to_fixture_specific_hsv(jdc1_al_hsv_backstage_floor, jdc1_group_count)
    jdc1_fixture_specific_frontstage_floor = jdc1_al_to_fixture_specific_hsv(jdc1_al_hsv_frontstage_floor, jdc1_group_count)

    # Write fixture specific values into artnet dummy
    AL_to_artnet = bmfl_fixture_specific_to_artnet(AL_to_artnet, bmfl_fixture_specific_ftruss, bmfl_group_truss_count, one_universe_size*0)
    AL_to_artnet = bmfl_fixture_specific_to_artnet(AL_to_artnet, bmfl_fixture_specific_m1truss, bmfl_group_truss_count, one_universe_size*2)
    AL_to_artnet = bmfl_fixture_specific_to_artnet(AL_to_artnet, bmfl_fixture_specific_btruss, bmfl_group_truss_count, one_universe_size*3)
    AL_to_artnet = bmfl_fixture_specific_to_artnet(AL_to_artnet, bmfl_fixture_specific_sidestage_front, bmfl_group_sidestage_count, one_universe_size*7)
    AL_to_artnet = jdc1_fixture_specific_to_artnet(AL_to_artnet, jdc1_fixture_specific_ftruss, jdc1_group_count, one_universe_size*9)
    AL_to_artnet = jdc1_fixture_specific_to_artnet(AL_to_artnet, jdc1_fixture_specific_m1truss, jdc1_group_count, one_universe_size*11)
    AL_to_artnet = jdc1_fixture_specific_to_artnet(AL_to_artnet, jdc1_fixture_specific_btruss, jdc1_group_count, one_universe_size*12)
    AL_to_artnet = jdc1_fixture_specific_to_artnet(AL_to_artnet, jdc1_fixture_specific_backstage_floor, jdc1_group_count, one_universe_size*13)
    AL_to_artnet = jdc1_fixture_specific_to_artnet(AL_to_artnet, jdc1_fixture_specific_frontstage_floor, jdc1_group_count, one_universe_size*14)
    #AL_to_artnet[:, one_universe_size*12:one_universe_size*13] = np.zeros((len, 512))

    return AL_to_artnet


#######################


def bmfl_al_to_fixture_specific_hsv(bmfl_al, count):

    # Source AL Extractor HSV BMFL
    # 01 = Intensity / Dim of Peak absolute
    # 02 = Col Hue of Peak
    # 03 = Col Sat of Peak
    #
    # Target BMFL Fixture Specific
    # |:
    # 1 = Dim, 2 = ColorHue, 3 = ColorSat, 4 = Pan, 5 = Tilt,
    # 6 = Shutter, 7 = Zoom, 8 = Iris
    # :|
    #
    one_bmfl_fixture_specific_size = 8
    size_group = bmfl_al.shape
    # update every frame, change this for use in matlab replacement for the whole array
    len = 1
    bmfl_fixture_specific = np.zeros((len, one_bmfl_fixture_specific_size * count))
    int_mean = np.zeros((len, 1))
    colhue_mean = np.zeros((len, 1))
    colsat_mean = np.zeros((len, 1))

    # Defaults / Fixed Values
    pan_mean_val = np.zeros((len, count)) + 0.5
    tilt_mean_val = np.zeros((len, count)) + 0.5
    shutter_val = np.zeros((len, count)) + 0.1875
    zoom_val = np.zeros((len, count)) + 0.5
    iris_val = np.zeros((len, count))
    temp4_val = np.zeros((len, count))
    temp5_val = np.zeros((len, count))

    for i in range(len):
        # get current values
        int_mean[i, 0] = bmfl_al[i, 0]
        colhue_mean[i, 0] = bmfl_al[i, 1]
        colsat_mean[i, 0] = bmfl_al[i, 2]

    al_parameter_struct = {'bmfl_al': bmfl_al, 'count': count, 'int_mean': int_mean,
                           'colhue_mean': colhue_mean, 'colsat_mean': colsat_mean,
                           'pan_mean_val': pan_mean_val, 'tilt_mean_val': tilt_mean_val,
                           'shutter_val': shutter_val, 'zoom_val': zoom_val,
                           'iris_val': iris_val, 'temp4_val': temp4_val, 'temp5_val': temp5_val}

    # get the AL-to-group values with the al_to_fixture_specific_hsv function
    groups_struct = al_to_fixture_specific_hsv(al_parameter_struct)

    groups_struct_temp = list(groups_struct.values())
    dim_group = np.array(groups_struct_temp[0])
    colhue_group = np.array(groups_struct_temp[1])
    colsat_group = np.array(groups_struct_temp[2])
    pan_group = np.array(groups_struct_temp[3])
    tilt_group = np.array(groups_struct_temp[4])
    shutter_group = np.array(groups_struct_temp[5])
    zoom_group = np.array(groups_struct_temp[6])
    iris_group = np.array(groups_struct_temp[7])

    # Target BMFL Fixture Specific
    # |:
    # 1 = Dim, 2 = ColorHue, 3 = ColorSat, 4 = Pan, 5 = Tilt,
    # 6 = Shutter, 7 = Zoom, 8 = Iris
    # :|
    #
    for n in range(count):
        bmfl_fixture_specific[:, n * 8 + 0] = dim_group[:, n]
        bmfl_fixture_specific[:, n * 8 + 1] = colhue_group[:, n]
        bmfl_fixture_specific[:, n * 8 + 2] = colsat_group[:, n]
        bmfl_fixture_specific[:, n * 8 + 3] = pan_group[:, n]
        bmfl_fixture_specific[:, n * 8 + 4] = tilt_group[:, n]
        ### Defaults / Fixed Values
        bmfl_fixture_specific[:, n * 8 + 5] = shutter_group[:, n]
        bmfl_fixture_specific[:, n * 8 + 6] = zoom_group[:, n]
        bmfl_fixture_specific[:, n * 8 + 7] = iris_group[:, n]

    return bmfl_fixture_specific


#######################


def jdc1_al_to_fixture_specific_hsv(jdc1_al, count):

    # Source AL Extractor HSV JDC1
    # 01 = Intensity of Peak absolute
    # 02 = Col Hue of Peak
    # 03 = Col Sat of Peak
    #
    # Target JDC1 group specific
    # |: 
    # 1 = Plate_Int, 2, Beam_Int, 
    # 3 = ColorHue, 4 = ColorSat, 5 = ColorVel,
    # 6 = Tilt, 7 = Plate_Shutter, 8 = Plate_Rate, 
    # 9 = Beam_Shutter, 10 = Beam_Rate
    # :|
    # 

    one_jdc1_fixture_specific_size = 10
    size_group = jdc1_al.shape
    # update every frame, change this for use in matlab replacement for the whole array
    len = 1
    
    jdc1_fixture_specific = np.zeros((len, one_jdc1_fixture_specific_size * count))
    int_mean = np.zeros((len, 1))
    colhue_mean = np.zeros((len, 1))
    colsat_mean = np.zeros((len, 1))
    ### Defaults / Fixed Values
    ### Change the default values !!
    pan_mean_val = np.zeros((len, count)) + 0.5
    tilt_mean_val = np.zeros((len, count)) + 0.5
    plate_shutter_val = np.zeros((len, count))
    plate_rate_val = np.zeros((len, count)) + 1.0
    beam_shutter_val = np.zeros((len, count)) + 0.0
    beam_rate_val = np.zeros((len, count)) + 1.0
    beam_mean_val = np.zeros((len, count)) + 0.0

    for i in range(len):
        # get current values
        int_mean[i, 0] = jdc1_al[i, 0]
        colhue_mean[i, 0] = jdc1_al[i, 1]
        colsat_mean[i, 0] = jdc1_al[i, 2]

    al_parameter_struct = {'jdc1_al': jdc1_al, 'count': count, 'int_mean': int_mean,
                       'colhue_mean': colhue_mean, 'colsat_mean': colsat_mean,
                       'pan_mean_val': pan_mean_val, 'tilt_mean_val': tilt_mean_val,
                       'plate_shutter_val': plate_shutter_val, 'plate_rate_val': plate_rate_val,
                       'beam_shutter_val': beam_shutter_val, 
                       'beam_rate_val': beam_rate_val, 'beam_mean_val': beam_mean_val}
    ###
    # get the AL-to-group values with the al_to_fixture_specific_hsv function 
    # consider removing this step cause there is maybe no need for extra
    # function
    groups_struct = al_to_fixture_specific_hsv(al_parameter_struct)

    groups_struct_temp = list(groups_struct.values())
    dim_group = np.array(groups_struct_temp[0])
    colhue_group = np.array(groups_struct_temp[1])
    colsat_group = np.array(groups_struct_temp[2])
    pan_group = np.array(groups_struct_temp[3])
    tilt_group = np.array(groups_struct_temp[4])
    plate_shutter_group = np.array(groups_struct_temp[5])
    plate_rate_group = np.array(groups_struct_temp[6])
    beam_shutter_group = np.array(groups_struct_temp[7])
    beam_rate_group = np.array(groups_struct_temp[8])
    beam_mean_group = np.array(groups_struct_temp[9])
    ####
    ####
    # Target JDC1 group specific
    # |: 
    # 1 = Plate_Int, 2, Beam_Int, 
    # 3 = ColorHue, 4 = ColorSat, 5 = ColorVel,
    # 6 = Tilt, 7 = Plate_Shutter, 8 = Plate_Rate, 
    # 9 = Beam_Shutter, 10 = Beam_Rate
    # :|
    # 
    for n in range(count):
        jdc1_fixture_specific[:,(n * 10 + 0)] = dim_group[:,n]
        jdc1_fixture_specific[:,(n * 10 + 1)] = beam_mean_group[:,n]
        jdc1_fixture_specific[:,(n * 10 + 2)] = colhue_group[:,n]
        jdc1_fixture_specific[:,(n * 10 + 3)] = colsat_group[:,n]
        jdc1_fixture_specific[:,(n * 10 + 4)] = dim_group[:,n]
        jdc1_fixture_specific[:,(n * 10 + 5)] = tilt_group[:,n]
        # Defaults / Fixed Values
        jdc1_fixture_specific[:,(n * 10 + 6)] = plate_shutter_group[:,n]
        jdc1_fixture_specific[:,(n * 10 + 7)] = plate_rate_group[:,n]
        jdc1_fixture_specific[:,(n * 10 + 8)] = beam_shutter_group[:,n]
        jdc1_fixture_specific[:,(n * 10 + 9)] = beam_rate_group[:,n]
        
    return jdc1_fixture_specific


#######################


def bmfl_al_to_fixture_specific_small(bmfl_al, bmfl_count):

    # Source AL Extractor Small
    # 01 = Position of Max Intensity
    # 02 = Intensity of Peak absolute
    # 03 = Slope of Peak Intensity
    # 04 = AF Peak Density
    # 05 = Col Hue Mean
    # 06 = Col Sat Mean
    # 07 = Pan Mean
    # 08 = Tilt Mean
    #
    # Target BMFL Fixture Specific
    # |: 
    # 1 = Dim, 2 = ColorHue, 3 = ColorSat, 4 = Pan, 5 = Tilt,
    # 6 = Shutter, 7 = Zoom, 8 = Iris
    # :|
    count = bmfl_count
    one_bmfl_fixture_specific_size = 8
    size_group = bmfl_al.shape
    len = size_group[0]
    bmfl_fixture_specific = np.zeros((len, one_bmfl_fixture_specific_size * count))
    colhue_mean = np.zeros((len, 1))
    colsat_mean = np.zeros((len, 1))
    pan_mean = np.zeros((len, 1))
    tilt_mean = np.zeros((len, 1))
    maxVal = np.zeros((len, 1))
    gradient = np.zeros((len, 1))
    posMaxInt = np.zeros((len, 1))
    approxAbsCountPeaks = np.zeros((len, 1))
    ### Defaults / Fixed Values
    shutter_val = np.zeros((len, count)) + 0.1875
    zoom_val = np.zeros((len, count)) + 0.5
    iris_val = np.zeros((len, count)) + 0.0
    temp4_val = np.zeros((len, count)) + 0.0
    temp5_val = np.zeros((len, count)) + 0.0

    for i in range(len):
        # get current values
        maxVal[i, 0] = bmfl_al[i, 1]
        gradient[i, 0] = bmfl_al[i, 2]
        posMaxInt[i, 0] = bmfl_al[i, 0]
        approxAbsCountPeaks[i, 0] = bmfl_al[i, 3]
        colhue_mean[i, 0] = bmfl_al[i, 4]
        colsat_mean[i, 0] = bmfl_al[i, 5]
        pan_mean[i, 0] = bmfl_al[i, 6]
        tilt_mean[i, 0] = bmfl_al[i, 7]
        
    al_parameter_struct = {'bmfl_al': bmfl_al, 'count': count, 'maxVal': maxVal,
                       'gradient': gradient, 'posMaxInt': posMaxInt,
                       'approxAbsCountPeaks': approxAbsCountPeaks, 'colhue_mean': colhue_mean,
                       'colsat_mean': colsat_mean, 'pan_mean': pan_mean,
                       'tilt_mean': tilt_mean, 'shutter_val': shutter_val, 'zoom_val': zoom_val,
                       'iris_val': iris_val, 'temp4_val': temp4_val, 'temp5_val': temp5_val}
    
    ###
    ### get the AL-to-group values with the al_to_fixture_specific_small function
    groups_struct = al_to_fixture_specific_small(al_parameter_struct)
    groups_struct_temp = list(groups_struct.values())
    dim_group = np.array(groups_struct_temp[0])
    colhue_group = np.array(groups_struct_temp[1])
    colsat_group = np.array(groups_struct_temp[2])
    pan_group = np.array(groups_struct_temp[3])
    tilt_group = np.array(groups_struct_temp[4])
    shutter_group = np.array(groups_struct_temp[5])
    zoom_group = np.array(groups_struct_temp[6])
    iris_group = np.array(groups_struct_temp[7])
    ###
    ###
    ### set the values to the fixture specific array
    # Target BMFL Fixture Specific
    # |: 
    # 1 = Dim, 2 = ColorHue, 3 = ColorSat, 4 = Pan, 5 = Tilt,
    # 6 = Shutter, 7 = Zoom, 8 = Iris
    # :|
    # 
    for n in range(count):
        bmfl_fixture_specific[:, n * 8 + 0] = dim_group[:, n]
        bmfl_fixture_specific[:, n * 8 + 1] = colhue_group[:, n]
        bmfl_fixture_specific[:, n * 8 + 2] = colsat_group[:, n]
        bmfl_fixture_specific[:, n * 8 + 3] = pan_group[:, n]
        bmfl_fixture_specific[:, n * 8 + 4] = tilt_group[:, n]
        ### Defaults / Fixed Values
        bmfl_fixture_specific[:, n * 8 + 5] = shutter_group[:, n]
        bmfl_fixture_specific[:, n * 8 + 6] = zoom_group[:, n]
        bmfl_fixture_specific[:, n * 8 + 7] = iris_group[:, n]

    return bmfl_fixture_specific


#######################


def jdc1_al_to_fixture_specific_small(jdc1_al, count):

    # Source AL Extractor Small JDC1
    # 01 = Plate Position of Max Intensity
    # 02 = Plate Intensity of Peak absolute
    # 03 = Plate Slope of Peak Intensity
    # 04 = Plate AF Peak Density
    # 05 = Beam Intensity Mean
    # 06 = Col Hue Mean
    # 07 = Col Sat Mean
    # 08 = Tilt Mean
    #
    # Target JDC1 group specific
    # |: 
    # 1 = Plate_Int, 2, Beam_Int, 
    # 3 = ColorHue, 4 = ColorSat, 5 = ColorVel,
    # 6 = Tilt, 7 = Plate_Shutter, 8 = Plate_Rate, 
    # 9 = Beam_Shutter, 10 = Beam_Rate
    # :|
    # 
    one_jdc1_fixture_specific_size = 10
    size_group = jdc1_al.shape
    len = size_group[0]
    jdc1_fixture_specific = np.zeros((len, one_jdc1_fixture_specific_size * count))
    maxVal = np.zeros((len, 1))
    gradient = np.zeros((len, 1))
    posMaxInt = np.zeros((len, 1))
    approxAbsCountPeaks = np.zeros((len, 1))
    beam_mean = np.zeros((len, 1))
    colhue_mean = np.zeros((len, 1))
    colsat_mean = np.zeros((len, 1))
    # pan mean is placeholder
    pan_mean = np.zeros((len, 1))
    tilt_mean = np.zeros((len, 1))
    temp1_val = np.zeros((len, 1))
    
    for i in range(len):
        # get current values
        maxVal[i, 0] = jdc1_al[i, 1]
        gradient[i, 0] = jdc1_al[i, 2]
        posMaxInt[i, 0] = jdc1_al[i, 0]
        approxAbsCountPeaks[i, 0] = jdc1_al[i, 3]
        beam_mean[i, 0] = jdc1_al[i, 4]
        colhue_mean[i, 0] = jdc1_al[i, 5]
        colsat_mean[i, 0] = jdc1_al[i, 6]
        tilt_mean[i, 0] = jdc1_al[i, 7]

    # Defaults / Fixed Values
    plate_shutter_val = np.zeros((len, count))
    plate_rate_val = np.ones((len, count))
    beam_shutter_val = np.zeros((len, count))
    beam_rate_val = np.ones((len, count))

    al_parameter_struct = {
        'jdc1_al': jdc1_al,
        'count': count,
        'maxVal': maxVal,
        'gradient': gradient,
        'posMaxInt': posMaxInt,
        'approxAbsCountPeaks': approxAbsCountPeaks,
        'colhue_mean': colhue_mean,
        'colsat_mean': colsat_mean,
        'pan_mean': tilt_mean,
        'tilt_mean': tilt_mean,
        'plate_shutter_val': plate_shutter_val,
        'plate_rate_val': plate_rate_val,
        'beam_shutter_val': beam_shutter_val,
        'beam_rate_val': beam_rate_val,
        'beam_mean': beam_mean,
        'temp1_val' : temp1_val
    }

    groups_struct = al_to_fixture_specific_small(al_parameter_struct)
    
    groups_struct_temp = list(groups_struct.values())
    dim_group = np.array(groups_struct_temp[0])
    colhue_group = np.array(groups_struct_temp[1])
    colsat_group = np.array(groups_struct_temp[2])
    pan_group = np.array(groups_struct_temp[3])
    tilt_group = np.array(groups_struct_temp[4])
    plate_shutter_group = np.array(groups_struct_temp[5])
    plate_rate_group = np.array(groups_struct_temp[6])
    beam_shutter_group = np.array(groups_struct_temp[7])
    beam_rate_group = np.array(groups_struct_temp[8])
    beam_mean_group = np.array(groups_struct_temp[9])

    for n in range(count):
        jdc1_fixture_specific[:, n * 10 + 0] = dim_group[:, n]
        jdc1_fixture_specific[:, n * 10 + 1] = beam_mean_group[:, n]
        jdc1_fixture_specific[:, n * 10 + 2] = colhue_group[:, n]
        jdc1_fixture_specific[:, n * 10 + 3] = colsat_group[:, n]
        jdc1_fixture_specific[:, n * 10 + 4] = dim_group[:, n]
        jdc1_fixture_specific[:, n * 10 + 5] = tilt_group[:, n]
        ### Defaults / Fixed Values
        jdc1_fixture_specific[:, n * 10 + 6] = plate_shutter_group[:, n]
        jdc1_fixture_specific[:, n * 10 + 7] = plate_rate_group[:, n] + 1.0
        jdc1_fixture_specific[:, n * 10 + 8] = beam_shutter_group[:, n]
        jdc1_fixture_specific[:, n * 10 + 9] = beam_rate_group[:, n] + 1.0
        
    return jdc1_fixture_specific


#######################


def bmfl_al_to_fixture_specific_large(bmfl_al, bmfl_count):
    
    # Source AL Extractor Large:
    # 01 = Position of Max Intensity
    # 02 = Intensity of Peak absolute
    # 03 = Slope of Peak Intensity
    # 04 = AF Peak Density
    # 05 = AF Peak Simularity
    # 06 = AF Pos 2nd Peak (if exists, otherwise Pos 1st)
    # 07 = AF Intensity of Minima absolute
    # 08 = AF Minima Simularity
    # 09 = Col Hue Lead
    # 10 = Col Sat Lead
    # 11 = Col Hue Back
    # 12 = Col Sat Back
    # 13 = Pos Max Left Pan
    # 14 = Val Max Left Pan
    # 15 = Pos Max Right Pan
    # 16 = Val Max Right Pan
    # 17 = Value of Max Tilt
    # 18 = Value of Min Tilt

    # Target BMFL Fixture Specific:
    # |:
    # 1 = Dim, 2 = ColorHue, 3 = ColorSat, 4 = Pan, 5 = Tilt,
    # 6 = Shutter, 7 = Zoom, 8 = Iris
    # :|
    
    count = bmfl_count
    one_bmfl_fixture_specific_size = 8
    size_group = bmfl_al.shape
    len = size_group[0]
    bmfl_fixture_specific = np.zeros((len, one_bmfl_fixture_specific_size * count))
    colhue_lead = np.zeros((len, 1))
    colhue_back = np.zeros((len, 1))
    colsat_lead = np.zeros((len, 1))
    colsat_back = np.zeros((len, 1))
    panmax_pos_left = np.zeros((len, 1))
    panmax_val_left = np.zeros((len, 1))
    panmax_pos_right = np.zeros((len, 1))
    panmax_val_right = np.zeros((len, 1))
    tiltmax_val = np.zeros((len, 1))
    tiltmin_val = np.zeros((len, 1))
    minVal = np.zeros((len, 1))
    maxVal = np.zeros((len, 1))
    gradient = np.zeros((len, 1))
    posMaxInt = np.zeros((len, 1))
    posMaxInt2nd = np.zeros((len, 1))
    approxAbsCountPeaks = np.zeros((len, 1))
    peakSimularity = np.zeros((len, 1))
    # Defaults / Fixed Values
    shutter_val = np.zeros((len, count)) + 0.1875
    zoom_val = np.zeros((len, count)) + 0.5
    iris_val = np.zeros((len, count)) + 0.0

    for i in range(len):
        # get current values
        minVal[i, 0] = bmfl_al[i, 6]
        maxVal[i, 0] = bmfl_al[i, 1]
        gradient[i, 0] = bmfl_al[i, 2]
        posMaxInt[i, 0] = bmfl_al[i, 0]
        posMaxInt2nd[i, 0] = bmfl_al[i, 5]
        approxAbsCountPeaks[i, 0] = bmfl_al[i, 3]
        peakSimularity[i, 0] = bmfl_al[i, 4]
        colhue_lead[i, 0] = bmfl_al[i, 8]
        colsat_lead[i, 0] = bmfl_al[i, 9]
        colhue_back[i, 0] = bmfl_al[i, 10]
        colsat_back[i, 0] = bmfl_al[i, 11]
        panmax_pos_left[i, 0] = bmfl_al[i, 12]
        panmax_val_left[i, 0] = bmfl_al[i, 13]
        panmax_pos_right[i, 0] = bmfl_al[i, 14]
        panmax_val_right[i, 0] = bmfl_al[i, 15]
        tiltmax_val[i, 0] = bmfl_al[i, 16]
        tiltmin_val[i, 0] = bmfl_al[i, 17]

    al_parameter_struct = {'bmfl_al': bmfl_al, 'count': count, 'minVal': minVal, 'maxVal': maxVal,
                   'gradient': gradient, 'posMaxInt': posMaxInt, 'posMaxInt2nd': posMaxInt2nd,
                   'approxAbsCountPeaks': approxAbsCountPeaks, 'peakSimularity': peakSimularity, 
                   'colhue_lead': colhue_lead, 'colsat_lead': colsat_lead, 'colhue_back': colhue_back,
                   'colsat_back': colsat_back, 'panmax_pos_left': panmax_pos_left, 'panmax_val_left': panmax_val_left,
                   'panmax_pos_right': panmax_pos_right, 'panmax_val_right': panmax_val_right, 
                   'tiltmax_val': tiltmax_val, 'tiltmin_val': tiltmin_val, 'shutter_val': shutter_val, 
                   'zoom_val': zoom_val, 'iris_val': iris_val}
    
    ###
    ### get the AL-to-group values with the al_to_fixture_specific_large function
    groups_struct = al_to_fixture_specific_large(al_parameter_struct)

    groups_struct_temp = list(groups_struct.values())
    dim_group = np.array(groups_struct_temp[0])
    colhue_group = np.array(groups_struct_temp[1])
    colsat_group = np.array(groups_struct_temp[2])
    pan_group = np.array(groups_struct_temp[3])
    tilt_group = np.array(groups_struct_temp[4])
    shutter_group = np.array(groups_struct_temp[5]) + 0.1875
    zoom_group = np.array(groups_struct_temp[6]) + 0.5
    iris_group = np.array(groups_struct_temp[7]) + 0.0
    ###
    ###
    ### set the values to the fixture specific array
    # Target BMFL Fixture Specific
    # |: 
    # 1 = Dim, 2 = ColorHue, 3 = ColorSat, 4 = Pan, 5 = Tilt,
    # 6 = Shutter, 7 = Zoom, 8 = Iris
    # :|
    # 
    for n in range(count):
        bmfl_fixture_specific[:, n * 8 + 0] = dim_group[:, n]
        bmfl_fixture_specific[:, n * 8 + 1] = colhue_group[:, n]
        bmfl_fixture_specific[:, n * 8 + 2] = colsat_group[:, n]
        bmfl_fixture_specific[:, n * 8 + 3] = pan_group[:, n]
        bmfl_fixture_specific[:, n * 8 + 4] = tilt_group[:, n]
        ### Defaults / Fixed Values
        bmfl_fixture_specific[:, n * 8 + 5] = shutter_group[:, n]
        bmfl_fixture_specific[:, n * 8 + 6] = zoom_group[:, n]
        bmfl_fixture_specific[:, n * 8 + 7] = iris_group[:, n]

    return bmfl_fixture_specific


#######################


def al_to_fixture_specific_hsv(al_parameter_struct):

    al_parameter_struct_temp = list(al_parameter_struct.values())
    al = np.array(al_parameter_struct_temp[0])
    count = np.array(al_parameter_struct_temp[1])
    int_mean = np.array(al_parameter_struct_temp[2])
    colhue_mean = np.array(al_parameter_struct_temp[3])
    colsat_mean = np.array(al_parameter_struct_temp[4])
    pan_mean = np.array(al_parameter_struct_temp[5])
    tilt_mean = np.array(al_parameter_struct_temp[6])
    temp1_val = np.array(al_parameter_struct_temp[7])
    temp2_val = np.array(al_parameter_struct_temp[8])
    temp3_val = np.array(al_parameter_struct_temp[9])
    temp4_val = np.array(al_parameter_struct_temp[10])
    temp5_val = np.array(al_parameter_struct_temp[11])

    size_group = al.shape
    len = size_group[0]
    dim_group = np.zeros((len, count))
    colhue_group = np.zeros((len, count))
    colsat_group = np.zeros((len, count))
    pan_group = np.zeros((len, count))
    tilt_group = np.zeros((len, count))
    temp1_group = np.zeros((len, count))
    temp2_group = np.zeros((len, count))
    temp3_group = np.zeros((len, count))
    temp4_group = np.zeros((len, count))
    temp5_group = np.zeros((len, count))

    for i in range(len):
        # set groups to the value of the mean
        # here in the loop for possible further development
        dim_group[i, :] = int_mean[i, 0]
        colhue_group[i, :] = colhue_mean[i, 0]
        colsat_group[i, :] = colsat_mean[i, 0]
        pan_group[i, :] = pan_mean[i, 0]
        tilt_group[i, :] = tilt_mean[i, 0]
        # Defaults / Fixed Values
        # do these also in the loop for some possible changes later
        temp1_group[i, :] = temp1_val[i, 0]
        temp2_group[i, :] = temp2_val[i, 0]
        temp3_group[i, :] = temp3_val[i, 0]
        temp4_group[i, :] = temp4_val[i, 0]
        temp5_group[i, :] = temp5_val[i, 0]

    groups_struct = {'dim_group': dim_group, 'colhue_group': colhue_group, 'colsat_group': colsat_group,
                    'pan_group': pan_group, 'tilt_group': tilt_group,
                    'temp1_group': temp1_group, 'temp2_group': temp2_group,
                    'temp3_group': temp3_group, 'temp4_group': temp4_group,
                    'temp5_group': temp5_group}

    return groups_struct


#######################


def al_to_fixture_specific_small(al_parameter_struct):

    al_parameter_struct_temp = list(al_parameter_struct.values())
    al = np.array(al_parameter_struct_temp[0])
    count = np.array(al_parameter_struct_temp[1])
    maxVal = np.array(al_parameter_struct_temp[2])
    gradient = np.array(al_parameter_struct_temp[3])
    posMaxInt = np.array(al_parameter_struct_temp[4])
    approxAbsCountPeaks = np.array(al_parameter_struct_temp[5])
    colhue_mean = np.array(al_parameter_struct_temp[6])
    colsat_mean = np.array(al_parameter_struct_temp[7])
    pan_mean = np.array(al_parameter_struct_temp[8])
    tilt_mean = np.array(al_parameter_struct_temp[9])
    temp1_val = np.array(al_parameter_struct_temp[10])
    temp2_val = np.array(al_parameter_struct_temp[11])
    temp3_val = np.array(al_parameter_struct_temp[12])
    temp4_val = np.array(al_parameter_struct_temp[13])
    temp5_val = np.array(al_parameter_struct_temp[14])
    #
    size_group = al.shape
    len = size_group[0]
    globalCurve_len = 513
    #
    dim_group = np.zeros((len, count))
    colhue_group = np.zeros((len, count))
    colsat_group = np.zeros((len, count))
    pan_group = np.zeros((len, count))
    tilt_group = np.zeros((len, count))
    temp1_group = np.zeros((len, count))
    temp2_group = np.zeros((len, count))
    temp3_group = np.zeros((len, count))
    temp4_group = np.zeros((len, count))
    temp5_group = np.zeros((len, count))

    for i in range(len):
        # get current values
        posMaxInt[i] = posMaxInt[i] * count
        posMaxInt[i] = int(np.round(np.clip(posMaxInt[i], 1, count))) - 1
        approxAbsCountPeaks[i] = np.round(approxAbsCountPeaks[i] * count)
        if approxAbsCountPeaks[i] > 1:
            peakDistance = np.floor((count - approxAbsCountPeaks[i])*1.0/approxAbsCountPeaks[i])
            tempDistance = peakDistance
            if tempDistance == 0.0:
                peakDistance = 1
        elif approxAbsCountPeaks[i] == 1:
            peakDistance = np.floor((count - approxAbsCountPeaks[i])/(approxAbsCountPeaks[i] * 1.0))
            tempDistance = peakDistance
            if tempDistance == 0.0:
                peakDistance = 1
        else:
            peakDistance = 0
        
        # set possible groups to the value of the mean
        # here in the loop for possible further development
        colhue_group[i, :] = colhue_mean[i]
        colsat_group[i, :] = colsat_mean[i]
        pan_group[i, :] = pan_mean[i]
        tilt_group[i, :] = tilt_mean[i]
        # Defaults / Fixed Values
        # do these also in the loop for some possible changes later
        temp1_group[i, :] = temp1_val[i]
        temp2_group[i, :] = temp2_val[i]
        temp3_group[i, :] = temp3_val[i]
        temp4_group[i, :] = temp4_val[i]
        temp5_group[i, :] = temp5_val[i]
        ###
        ###
        # the hard-coded factors / parameters should be considered during
        # testing on the visualiser
        ######
        gradient_temp = np.clip(gradient[i,0], 0, 0.99)
        distToMin = round((1.0 - gradient_temp) * count / 1.5)
        ######
        localCurveLength1st = distToMin * 2 + 1
        
        # interpolate values linear - consider a using a sin envelope
        deltaIntensityStep = (maxVal[i,0] / localCurveLength1st) * 1.85
        localCurve1st = np.zeros(localCurveLength1st)
        #print(deltaIntensityStep)
        
        # generate local curve - set linear interpolate values left and right of the max value in the middle
        for p in range(1,(localCurveLength1st + 1)):
            #print(p)
            if p < (distToMin + 1):
                localCurve1st[p - 1] = deltaIntensityStep * (p)
            elif p > (distToMin + 1):
                localCurve1st[p - 1] = deltaIntensityStep * (localCurveLength1st - p + 1)
            else:
                localCurve1st[p - 1] = maxVal[i,0]

        globalCurve = np.zeros(globalCurve_len)
        globalCurve_cut = np.zeros(globalCurve_len * 2)

        # position first curve
        if ((posMaxInt[i,0] - distToMin) == 0):
            globalCurve[:localCurveLength1st] = localCurve1st
        elif ((posMaxInt[i,0] - distToMin) > 0):
            globalCurve[int(posMaxInt[i,0] - distToMin):int(posMaxInt[i,0] - distToMin) + localCurveLength1st] = localCurve1st
        else:
            shifter = int(abs(posMaxInt[i,0] - distToMin))
            globalCurve[:(localCurveLength1st - shifter)] = localCurve1st[shifter:]

        # write the other curves in case that they are existing
        
        # for the case that approxAbsCountPeaks[i]) == 1 only one localCurve is needed
        if (approxAbsCountPeaks[i]) < 1:
            # make them all equal
            globalCurve[:] = maxVal[i,0]
        elif (approxAbsCountPeaks[i]) > 1:
            if int(np.ceil(peakDistance / 2)) > distToMin:
                # don't cut the curve
                for q in range(1,int(approxAbsCountPeaks[i])):
                    posMaxInt_2nd_plus = int(posMaxInt[i,0] + peakDistance*q)
                    globalCurve[int(posMaxInt_2nd_plus - distToMin):(int(posMaxInt_2nd_plus - distToMin) + localCurveLength1st)] = localCurve1st
                    #print('local curves are not cutted')
            else:
                # cut the curve
                distToMin_cut = int(np.floor(peakDistance / 2))
                localCurveLength_cut = distToMin_cut + distToMin + 1
                distToMin_delta = distToMin - distToMin_cut
                localCurve1st_cut = np.zeros(localCurveLength_cut)
                # cut on the left side
                localCurve1st_cut = localCurve1st[distToMin_delta:]
                for q in range(1,int(approxAbsCountPeaks[i])):
                    posMaxInt_2nd_plus = int(posMaxInt[i,0] + peakDistance*q)
                    globalCurve[int(posMaxInt_2nd_plus - distToMin_cut):(int(posMaxInt_2nd_plus - distToMin_cut) + localCurveLength_cut)] = localCurve1st_cut
                    #print('local curves are cutted')

        # cut global curve to the right length
        globalCurve = globalCurve[:count]
        dim_group[i,:] = globalCurve


    groups_struct = {'dim_group': dim_group, 'colhue_group': colhue_group, 'colsat_group': colsat_group,
                    'pan_group': pan_group, 'tilt_group': tilt_group,
                    'temp1_group': temp1_group, 'temp2_group': temp2_group,
                    'temp3_group': temp3_group, 'temp4_group': temp4_group,
                    'temp5_group': temp5_group}

    return groups_struct



#######################

def al_to_fixture_specific_large(al_parameter_struct):

    al_parameter_struct_temp = list(al_parameter_struct.values())
    al = np.array(al_parameter_struct_temp[0])
    count = np.array(al_parameter_struct_temp[1])
    minVal = np.array(al_parameter_struct_temp[2])
    maxVal = np.array(al_parameter_struct_temp[3])
    gradient = np.array(al_parameter_struct_temp[4])
    posMaxInt = np.array(al_parameter_struct_temp[5])
    posMaxInt2nd = np.array(al_parameter_struct_temp[6])
    approxAbsCountPeaks = np.array(al_parameter_struct_temp[7])
    peakSimularity = np.array(al_parameter_struct_temp[8])
    colhue_lead = np.array(al_parameter_struct_temp[9])
    colsat_lead = np.array(al_parameter_struct_temp[10])
    colhue_back = np.array(al_parameter_struct_temp[11])
    colsat_back = np.array(al_parameter_struct_temp[12])
    panmax_pos_left = np.array(al_parameter_struct_temp[13])
    panmax_val_left = np.array(al_parameter_struct_temp[14])
    panmax_pos_right = np.array(al_parameter_struct_temp[15])
    panmax_val_right = np.array(al_parameter_struct_temp[16])
    tiltmax_val = np.array(al_parameter_struct_temp[17])
    tiltmin_val = np.array(al_parameter_struct_temp[18])
    shutter_val = np.array(al_parameter_struct_temp[19])
    zoom_val = np.array(al_parameter_struct_temp[20])
    iris_val = np.array(al_parameter_struct_temp[21])

    ### temporary small mode to large mode backconverter
    colhue_mean = colhue_lead
    colsat_mean = colsat_lead
    pan_mean = panmax_val_left - (panmax_val_left - panmax_val_right)/2
    tilt_mean = tiltmax_val - (tiltmax_val - tiltmin_val)/2
    ### temporary small mode to large mode backconverter - end
    #print(maxVal)
    #
    size_group = al.shape
    len = size_group[0]
    globalCurve_len = 513
    #
    dim_group = np.zeros((len, count))
    colhue_group = np.zeros((len, count))
    colsat_group = np.zeros((len, count))
    pan_group = np.zeros((len, count))
    tilt_group = np.zeros((len, count))
    temp1_group = np.zeros((len, count))
    temp2_group = np.zeros((len, count))
    temp3_group = np.zeros((len, count))
    temp4_group = np.zeros((len, count))
    temp5_group = np.zeros((len, count))

    for i in range(len):
        # get current values
        posMaxInt[i] = posMaxInt[i] * count
        posMaxInt[i] = int(np.round(np.clip(posMaxInt[i], 0, count)))
        approxAbsCountPeaks[i] = np.round(approxAbsCountPeaks[i] * count)
        if approxAbsCountPeaks[i] > 1:
            peakDistance = np.floor((count - approxAbsCountPeaks[i])*1.2/approxAbsCountPeaks[i])
            tempDistance = peakDistance
            if tempDistance == 0.0:
                peakDistance = 1
        elif approxAbsCountPeaks[i] == 1:
            peakDistance = np.floor((count - approxAbsCountPeaks[i])/(approxAbsCountPeaks[i] * 1.0))
            tempDistance = peakDistance
            if tempDistance == 0.0:
                peakDistance = 1
        else:
            peakDistance = 0
        
        # set possible groups to the value of the mean
        # here in the loop for possible further development
        colhue_group[i, :] = colhue_mean[i]
        colsat_group[i, :] = colsat_mean[i]
        pan_group[i, :] = pan_mean[i]
        tilt_group[i, :] = tilt_mean[i]
        # Defaults / Fixed Values
        ###
        ###
        # the hard-coded factors / parameters should be considered during
        # testing on the visualiser
        ######
        gradient_temp = np.clip(gradient[i,0], 0, 0.99)
        distToMin = round((1.0 - gradient_temp) * count / 1.5)
        ######
        localCurveLength1st = distToMin * 2 + 1
        
        # interpolate values linear - consider a using a sin envelope
        deltaIntensityStep = (maxVal[i,0] / localCurveLength1st) * 1.85
        localCurve1st = np.zeros(localCurveLength1st)
        
        # generate local curve - set linear interpolate values left and right of the max value in the middle
        for p in range(1,(localCurveLength1st + 1)):
            #print(p)
            if p < (distToMin + 1):
                localCurve1st[p - 1] = deltaIntensityStep * (p)
            elif p > (distToMin + 1):
                localCurve1st[p - 1] = deltaIntensityStep * (localCurveLength1st - p + 1)
            else:
                localCurve1st[p - 1] = maxVal[i,0]

        globalCurve = np.zeros(globalCurve_len)
        globalCurve_cut = np.zeros(globalCurve_len * 2)

        # position first curve
        if ((posMaxInt[i,0] - distToMin) == 0):
            globalCurve[:localCurveLength1st] = localCurve1st
        elif ((posMaxInt[i,0] - distToMin) > 0):
            globalCurve[int(posMaxInt[i,0] - distToMin):int(posMaxInt[i,0] - distToMin) + localCurveLength1st] = localCurve1st
        else:
            shifter = int(abs(posMaxInt[i,0] - distToMin))
            globalCurve[:(localCurveLength1st - shifter)] = localCurve1st[shifter:]

        # write the other curves in case that they are existing
        
        # for the case that approxAbsCountPeaks[i]) == 1 only one localCurve is needed
        if (approxAbsCountPeaks[i]) < 1:
            # make them all equal
            globalCurve[:] = maxVal[i,0]
        elif (approxAbsCountPeaks[i]) > 1:
            if int(np.ceil(peakDistance / 2)) > distToMin:
                # don't cut the curve
                for q in range(1,int(approxAbsCountPeaks[i])):
                    posMaxInt_2nd_plus = int(posMaxInt[i,0] + peakDistance*q)
                    globalCurve[int(posMaxInt_2nd_plus - distToMin):(int(posMaxInt_2nd_plus - distToMin) + localCurveLength1st)] = localCurve1st
                    #print('local curves are not cutted')
            else:
                # cut the curve
                distToMin_cut = int(np.floor(peakDistance / 2))
                localCurveLength_cut = distToMin_cut + distToMin + 1
                distToMin_delta = distToMin - distToMin_cut
                localCurve1st_cut = np.zeros(localCurveLength_cut)
                # cut on the left side
                localCurve1st_cut = localCurve1st[distToMin_delta:]
                for q in range(1,int(approxAbsCountPeaks[i])):
                    #print(q)
                    posMaxInt_2nd_plus = int(posMaxInt[i,0] + peakDistance*q)
                    globalCurve[int(posMaxInt_2nd_plus - distToMin_cut):(int(posMaxInt_2nd_plus - distToMin_cut) + localCurveLength_cut)] = localCurve1st_cut
                    #print('local curves are cutted')

        # cut global curve to the right length
        globalCurve = globalCurve[:count]
        dim_group[i,:] = globalCurve


    groups_struct = {'dim_group': dim_group, 'colhue_group': colhue_group, 'colsat_group': colsat_group,
                    'pan_group': pan_group, 'tilt_group': tilt_group,
                    'temp1_group': temp1_group, 'temp2_group': temp2_group,
                    'temp3_group': temp3_group, 'temp4_group': temp4_group,
                    'temp5_group': temp5_group}

    return groups_struct


#######################


def bmfl_fixture_specific_to_artnet(AL_to_artnet, bmfl_fixture_specific, bmfl_count, startaddress):
    len = AL_to_artnet.shape[0]
    one_bmfl_size = 48
    bmfl_region_artnet = AL_to_artnet[:, startaddress:(startaddress + bmfl_count*one_bmfl_size)]

    # Features BMFL #01
    # |: 
    # 1 = Dim, 2 = ColorHue, 3 = ColorSat, 4 = Pan, 5 = Tilt,
    # 6 = Shutter, 7 = Zoom, 8 = Iris
    # :|

    for i in range(bmfl_count):
        # counter DIM source
        j = i * 8
        j -= 8
        # counter DIM target
        k = i
        k_msb = k * one_bmfl_size - 2
        k_lsb = k * one_bmfl_size - 1
        # get extractor array DIM
        bmfl_dim = bmfl_fixture_specific[:, j]
        # set to MSB / LSB
        dim_temp = bmfl_dim * 256
        dim_msb = np.floor(dim_temp) / 256
        dim_lsb = np.mod(dim_temp, 1)
        bmfl_region_artnet[:, k_msb] = dim_msb
        bmfl_region_artnet[:, k_lsb] = dim_lsb

        # counter Color source
        j = i * 8
        n = j - 7
        l = j - 6
        # color temp store
        color_temp_hsv = np.zeros((len, 3))
        color_temp_hsv[:, 0] = bmfl_fixture_specific[:, n]
        color_temp_hsv[:, 1] = bmfl_fixture_specific[:, l]
        color_temp_hsv[:, 2] = bmfl_dim
        (r, g, b) = (colorsys.hsv_to_rgb(color_temp_hsv[:, 0], color_temp_hsv[:, 1], color_temp_hsv[:, 2]))
        cyan = 1 - r
        mag = 1 - g
        yell = 1 - b
        #cyan = 1 - color_temp_rgb[:, 0]
        #mag = 1 - color_temp_rgb[:, 1]
        #yell = 1 - color_temp_rgb[:, 2]
        # counter Color target
        k = i
        k_cyan = k * one_bmfl_size - 38
        k_mag = k * one_bmfl_size - 37
        k_yell = k * one_bmfl_size - 36
        k_cto = k * one_bmfl_size - 35
        #
        bmfl_region_artnet[:, k_cyan] = cyan
        bmfl_region_artnet[:, k_mag] = mag
        bmfl_region_artnet[:, k_yell] = yell
        # no cto value, because this was added in the forward calc
        bmfl_region_artnet[:, k_cto] = 0

        # counter PAN source
        j = i * 8
        j -= 5
        bmfl_pan = bmfl_fixture_specific[:, j]
        # counter PAN target
        k = i
        k_msb = k * one_bmfl_size - 48
        k_lsb = k * one_bmfl_size - 47
        # set to MSB / LSB
        pan_temp = bmfl_pan * 256
        pan_msb = np.floor(pan_temp) / 256
        pan_lsb = np.mod(pan_temp, 1)
        bmfl_region_artnet[:, k_msb] = pan_msb
        bmfl_region_artnet[:, k_lsb] = pan_lsb

        # counter TILT source
        j = i * 8
        j -= 4
        bmfl_tilt = bmfl_fixture_specific[:, j]
        # counter TILT target
        k = i
        k_msb = k * one_bmfl_size - 46
        k_lsb = k * one_bmfl_size - 45
        # set to MSB / LSB
        tilt_temp = bmfl_tilt * 256
        tilt_msb = np.floor(tilt_temp) / 256
        tilt_lsb = np.mod(tilt_temp, 1)
        bmfl_region_artnet[:, k_msb] = tilt_msb
        bmfl_region_artnet[:, k_lsb] = tilt_lsb

        # counter Iris source
        j = i * 8
        j -= 1
        bmfl_iris = bmfl_fixture_specific[:, j]
        # counter Zoom source
        j = i * 8
        j -= 2
        bmfl_zoom = bmfl_fixture_specific[:, j]
        # counter Shutter source
        j = i * 8
        j -= 3
        bmfl_shutter = bmfl_fixture_specific[:, j]
        # counter Iris target
        k = i
        k_msb = k * one_bmfl_size - 21
        k_lsb = k * one_bmfl_size - 20
        bmfl_region_artnet[:, k_msb] = bmfl_iris
        bmfl_region_artnet[:, k_lsb] = 0
        # counter Zoom taget
        k = i
        k_msb = k * one_bmfl_size - 19
        k_lsb = k * one_bmfl_size - 18
        bmfl_region_artnet[:, k_msb] = bmfl_zoom
        bmfl_region_artnet[:, k_lsb] = 0
        # counter Shutter target
        k = i
        k = k * one_bmfl_size - 3
        bmfl_region_artnet[:, k] = bmfl_shutter

    AL_to_artnet[:, startaddress:(startaddress + bmfl_count*one_bmfl_size)] = bmfl_region_artnet

    return AL_to_artnet


#######################


def jdc1_fixture_specific_to_artnet(AL_to_artnet, jdc1_fixture_specific, jdc1_count, startaddress):
    len = AL_to_artnet.shape[0]
    one_jdc1_size = 62
    jdc1_region_artnet = AL_to_artnet[:, startaddress:(startaddress + jdc1_count*one_jdc1_size)]
    
    # Feature Extractor JDC1 #01
    # |:
    # 1 = Plate_Int, 2, Beam_Int,
    # 3 = ColorHue, 4 = ColorSat, 5 = ColorVel,
    # 6 = Tilt, 7 = Plate_Shutter, 8 = Plate_Rate,
    # 9 = Beam_Shutter, 10 = Beam_Rate
    # :|
    
    for i in range(jdc1_count):
        # counter Plate_Int source
        j = i * 10
        j -= 10
        # counter Plate_Int target
        k = i
        k = k * one_jdc1_size - 55
        jdc1_plate_int = jdc1_fixture_specific[:, j]
        jdc1_region_artnet[:, k] = jdc1_plate_int
        # counter Beam_Int source
        j = i * 10
        j -= 9
        # counter Beam_Int target
        k = i
        k = k * one_jdc1_size - 60
        jdc1_beam_int = jdc1_fixture_specific[:, j]
        jdc1_region_artnet[:, k] = jdc1_beam_int
        
        # counter Color source
        j = i * 10
        jh = j - 8
        js = j - 7
        jv = j - 6
        # color temp store
        #color_temp_rgb = np.zeros((len, 3))
        color_temp_hsv = np.zeros((len, 3))
        color_temp_hsv[:, 0] = jdc1_fixture_specific[:, jh]
        color_temp_hsv[:, 1] = jdc1_fixture_specific[:, js]
        color_temp_hsv[:, 2] = jdc1_fixture_specific[:, jv]
        (r, g, b) = (colorsys.hsv_to_rgb(color_temp_hsv[:, 0], color_temp_hsv[:, 1], color_temp_hsv[:, 2]))
        
        # counter Color target
        # counter Plate first Pixel source
        k = i
        k_red = k * one_jdc1_size - 48
        k_green = k * one_jdc1_size - 47
        k_blue = k * one_jdc1_size - 46
        #
            
        for n in range(1, 13):
            v = n * 3
            h = v - 2
            s = v - 1
            jdc1_region_artnet[:, (k_red + v - 3)] = r
            jdc1_region_artnet[:, (k_green + v - 3)] = g
            jdc1_region_artnet[:, (k_blue + v - 3)] = b
            
        # DK2020 Mode Chan 12, 13, 14 needs 100%
        k = i
        k_plate_int_r = k * one_jdc1_size - 51
        k_plate_int_g = k * one_jdc1_size - 50
        k_plate_int_b = k * one_jdc1_size - 49
        jdc1_region_artnet[:, k_plate_int_r] = r
        jdc1_region_artnet[:, k_plate_int_g] = g
        jdc1_region_artnet[:, k_plate_int_b] = b

        # counter TILT source
        j = i * 10
        j -= 5
        jdc1_tilt = jdc1_fixture_specific[:, j]
        # counter TILT target
        k = i
        k_msb = k * one_jdc1_size - 62
        k_lsb = k * one_jdc1_size - 61
        # set to MSB / LSB
        tilt_temp = jdc1_tilt * 256
        tilt_msb = np.floor(tilt_temp)
        tilt_msb = tilt_msb / 256
        tilt_lsb = np.mod(tilt_temp, 1)
        jdc1_region_artnet[:, k_msb] = tilt_msb
        jdc1_region_artnet[:, k_lsb] = tilt_lsb

        # counter Plate_Shutter source
        j = i * 10
        j -= 4
        # counter Plate_Shutter target
        k = i
        k = k * one_jdc1_size - 52
        jdc1_region_artnet[:, k] = jdc1_fixture_specific[:, j]

        # counter Plate_Rate source
        j = i * 10
        j -= 3
        # counter Plate_Rate target
        k = i
        k = k * one_jdc1_size - 53
        jdc1_region_artnet[:, k] = jdc1_fixture_specific[:, j]

        # counter Beam_Shutter source
        j = i * 10
        j -= 2
        # counter Beam_Shutter target
        k = i
        k = k * one_jdc1_size - 57
        jdc1_region_artnet[:, k] = jdc1_fixture_specific[:, j]

        # counter Beam_Rate source
        j = i * 10
        j -= 1
        # counter Beam_Rate target
        k = i
        k = k * one_jdc1_size - 58
        jdc1_region_artnet[:, k] = jdc1_fixture_specific[:, j]
        
    AL_to_artnet[:, startaddress:(startaddress + jdc1_count*one_jdc1_size)] = jdc1_region_artnet

    return AL_to_artnet


#######################


def bound(x, bl, bu):
    # return bounded value clipped between bl and bu
    y = min(max(x, bl), bu)
    return y


#######################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='al_standard')
    parser.add_argument('--input_dir', type=str, default='artnet_data')

    base_args = parser.parse_args()

    config = pyhocon.ConfigFactory.parse_file("al.conf")[base_args.config]
    config.name = base_args.config
    config.input_dir = base_args.input_dir

    # Load JSON file

    with open('al_file.json', 'r') as f:
        data = json.load(f)
    al_dummy_temp = np.array(data)

    al_dummy = np.zeros((al_dummy_temp.shape[0], config.len_al))

    for i in range (al_dummy_temp.shape[0]):
        for j in range (config.len_al) :
            al_dummy[i,j] = al_dummy_temp[i,j] / 256
        ### Main Loop
        artnet = features_72params_to_artnet_DKv03(al_dummy[i,:],config)
        ###

    # Write JSON file to disk
    #
    # Convert the numpy array to a list
    artnet_list = artnet.tolist()
    with open('artnet_file.json', 'w') as f:
        json.dump(artnet_list, f)
