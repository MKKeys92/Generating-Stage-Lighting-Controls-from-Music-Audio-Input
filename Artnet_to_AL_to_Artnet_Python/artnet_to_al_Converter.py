import numpy as np
import json
import colorsys
import argparse
import pyhocon


#######################


def extract_features_Artnet_to_72params_v02(artnet_dummy,config):

    len_16uni = config.len_16uni
    len_al_all = config.len_al_all
    one_universe_size = config.one_universe_size
    bmfl_group_truss_count = config.bmfl_group_truss_count
    bmfl_group_sidestage_count = config.bmfl_group_sidestage_count
    bmfl_group_floor_count = config.bmfl_group_floor_count
    jdc_group_count = config.jdc_group_count
    one_bmfl_extract_size = config.one_bmfl_extract_size
    one_jdc1_extract_size = config.one_jdc1_extract_size
    one_bmfl_size = config.one_bmfl_size

    features_DKv01_1800 = np.zeros((1,len_al_all))

    ### extract groups raws
    # Feature Extractor BMFL #01
    # |: 
    # 1 = Dim, 2 = ColorHue, 3 = ColorSat, 4 = Pan, 5 = Tilt,
    # 6 = Shutter, 7 = Zoom, 8 = Iris
    # :|
    bmfl_extract_group_ftruss = bmfl_extract_from_artnet(artnet_dummy,bmfl_group_truss_count,one_universe_size*0,config)
    bmfl_extract_group_m1truss = bmfl_extract_from_artnet(artnet_dummy,bmfl_group_truss_count,one_universe_size*1,config)
    bmfl_extract_group_m2truss = bmfl_extract_from_artnet(artnet_dummy,bmfl_group_truss_count,one_universe_size*2,config)
    bmfl_extract_group_btruss = bmfl_extract_from_artnet(artnet_dummy,bmfl_group_truss_count,one_universe_size*3,config)
    bmfl_extract_group_sidestage_front = bmfl_extract_from_artnet(artnet_dummy,bmfl_group_sidestage_count,one_universe_size*4,config)
    bmfl_extract_group_sidestage_back = bmfl_extract_from_artnet(artnet_dummy,bmfl_group_sidestage_count,one_universe_size*5,config)
    bmfl_extract_group_frontstage_floor = bmfl_extract_from_artnet(artnet_dummy,bmfl_group_floor_count,one_universe_size*6,config)
    bmfl_extract_group_sidestage_wings = bmfl_extract_from_artnet(artnet_dummy,bmfl_group_sidestage_count,one_universe_size*7,config)
    bmfl_extract_group_foh = bmfl_extract_from_artnet(artnet_dummy,bmfl_group_sidestage_count,one_universe_size*8,config)
    #
    # Feature Extractor JDC1 #01
    # |: 
    # 1 = Plate_Int, 2, Beam_Int, 
    # 3 = ColorHue, 4 = ColorSat, 5 = ColorVel,
    # 6 = Tilt, 7 = Plate_Shutter, 8 = Plate_Rate, 
    # 9 = Beam_Shutter, 10 = Beam_Rate
    # :|
    jdc1_extract_group_ftruss = jdc1_extract_from_artnet(artnet_dummy,jdc_group_count,one_universe_size*9,config)
    jdc1_extract_group_m1truss = jdc1_extract_from_artnet(artnet_dummy,jdc_group_count,one_universe_size*10,config)
    jdc1_extract_group_m2truss = jdc1_extract_from_artnet(artnet_dummy,jdc_group_count,one_universe_size*11,config)
    jdc1_extract_group_btruss = jdc1_extract_from_artnet(artnet_dummy,jdc_group_count,one_universe_size*12,config)
    jdc1_extract_group_frontstage_floor = jdc1_extract_from_artnet(artnet_dummy,jdc_group_count,one_universe_size*13,config)
    jdc1_extract_group_backstage_floor = jdc1_extract_from_artnet(artnet_dummy,jdc_group_count,one_universe_size*14,config)
    #
    ####
    ####
    #### Get Abstraction Layer Values ####
    #
    bmfl_al_large_ftruss = bmfl_al_extractor_large(bmfl_extract_group_ftruss,bmfl_group_truss_count,config)
    jdc1_al_small_ftruss = jdc1_al_extractor_small(jdc1_extract_group_ftruss,jdc_group_count,config)
    bmfl_al_small_m1truss = bmfl_al_extractor_small(bmfl_extract_group_m1truss,bmfl_group_truss_count,config)
    bmfl_al_small_m2truss = bmfl_al_extractor_small(bmfl_extract_group_m2truss,bmfl_group_truss_count,config)
    jdc1_al_small_m1truss = jdc1_al_extractor_small(jdc1_extract_group_m1truss,jdc_group_count,config)
    jdc1_al_small_m2truss = jdc1_al_extractor_small(jdc1_extract_group_m2truss,jdc_group_count,config)
    bmfl_al_large_btruss = bmfl_al_extractor_large(bmfl_extract_group_btruss,bmfl_group_truss_count,config)
    jdc1_al_hsv_btruss = jdc1_al_extractor_hsv(jdc1_extract_group_btruss,jdc_group_count,config)
    jdc1_al_hsv_backstage_floor = jdc1_al_extractor_hsv(jdc1_extract_group_backstage_floor,jdc_group_count,config)
    jdc1_al_hsv_frontstage_floor = jdc1_al_extractor_hsv(jdc1_extract_group_frontstage_floor,jdc_group_count,config)
    bmfl_al_hsv_sidestage_front = bmfl_al_extractor_hsv(bmfl_extract_group_sidestage_front,bmfl_group_sidestage_count,config)
    bmfl_al_hsv_sidestage_back = bmfl_al_extractor_hsv(bmfl_extract_group_sidestage_back,bmfl_group_sidestage_count,config)
    bmfl_al_hsv_sidestage_wings = bmfl_al_extractor_hsv(bmfl_extract_group_sidestage_wings,bmfl_group_sidestage_count,config)
    bmfl_al_hsv_foh = bmfl_al_extractor_hsv(bmfl_extract_group_foh,bmfl_group_sidestage_count,config)
    bmfl_al_hsv_frontstage_floor = bmfl_al_extractor_hsv(bmfl_extract_group_frontstage_floor,bmfl_group_floor_count,config)
    #
    # Set Abstraction Layer to output
    # Consider to merge the m1 and m2 truss values..
    # print(bmfl_al_large_ftruss.shape)
    features_DKv01_1800[:, 0:18] = bmfl_al_large_ftruss
    features_DKv01_1800[:, 18:26] = jdc1_al_small_ftruss
    features_DKv01_1800[:, 26:34] = bmfl_al_small_m2truss
    features_DKv01_1800[:, 34:42] = jdc1_al_small_m1truss
    features_DKv01_1800[:, 42:60] = bmfl_al_large_btruss
    features_DKv01_1800[:, 60:63] = jdc1_al_hsv_btruss
    features_DKv01_1800[:, 63:66] = jdc1_al_hsv_backstage_floor
    features_DKv01_1800[:, 66:69] = jdc1_al_hsv_frontstage_floor
    features_DKv01_1800[:, 69:72] = bmfl_al_hsv_sidestage_front

    max1 = np.max(features_DKv01_1800, axis=0)
    max2 = np.max(max1)

    """
    if max2 <= 1.0 :
        print('all values below 1.0')
    else:
        print('a value over 1.0 appeared')
    """

    return features_DKv01_1800


#######################


def bmfl_al_extractor_hsv(bmfl_group, bmfl_count, config):
    # Source
    # Feature Extractor BMFL #01 for each fixture
    # |:
    # 1 = Dim, 2 = ColorHue, 3 = ColorSat, 4 = Pan, 5 = Tilt,
    # 6 = Shutter, 7 = Zoom, 8 = Iris
    # :|
    #
    # Target AL Extractor HSV
    # 01 = Intensity of Peak absolute
    # 02 = Col Hue of Peak
    # 03 = Col Sat of Peak
    one_bmfl_extract_size = config.one_bmfl_extract_size
    len = bmfl_group.shape[0]
    bmfl_al_hsv = np.zeros((len, 3))

    dim_bmfl = param_extractor(bmfl_group, bmfl_count, one_bmfl_extract_size, 0)
    colhue_bmfl = param_extractor(bmfl_group, bmfl_count, one_bmfl_extract_size, 1)
    colsat_bmfl = param_extractor(bmfl_group, bmfl_count, one_bmfl_extract_size, 2)

    maxvel_dim_bmfl, _ = find_maxvel_maxpos(dim_bmfl)
    bmfl_al_hsv[:, 0] = maxvel_dim_bmfl

    for i in range(len):
        maxpos = np.argmax(dim_bmfl[i, :])
        bmfl_al_hsv[i, 1] = colhue_bmfl[i, maxpos]
        bmfl_al_hsv[i, 2] = colsat_bmfl[i, maxpos]

    return bmfl_al_hsv


#######################


def jdc1_al_extractor_hsv(jdc1_group, jdc1_count, config):
    # Source
    # Feature Extractor JDC1 #01 for each fixture
    # |
    # 1 = Plate_Int, 2, Beam_Int,
    # 3 = ColorHue, 4 = ColorSat, 5 = ColorVel,
    # 6 = Tilt, 7 = Plate_Shutter, 8 = Plate_Rate,
    # 9 = Beam_Shutter, 10 = Beam_Rate
    # :|
    #
    # Target AL Extractor HSV
    # 01 = Intensity of Peak absolute
    # 02 = Col Hue of Peak
    # 03 = Col Sat of Peak
    one_jdc1_extract_size = config.one_jdc1_extract_size
    len = jdc1_group.shape[0]
    jdc1_al_hsv = np.zeros((len, 3))

    int_jdc1_plate = param_extractor(jdc1_group, jdc1_count, one_jdc1_extract_size, 0)
    colhue_jdc1_plate = param_extractor(jdc1_group, jdc1_count, one_jdc1_extract_size, 2)
    colsat_jdc1_plate = param_extractor(jdc1_group, jdc1_count, one_jdc1_extract_size, 3)

    maxvel_dim_jdc1, _ = find_maxvel_maxpos(int_jdc1_plate)
    jdc1_al_hsv[:, 0] = maxvel_dim_jdc1

    # When AL features are ok, reduce to one loop
    # Maxima absolute
    for i in range(len):
        maxpos = np.argmax(int_jdc1_plate[i, :])
        jdc1_al_hsv[i, 1] = colhue_jdc1_plate[i, maxpos]
        jdc1_al_hsv[i, 2] = colsat_jdc1_plate[i, maxpos]

    return jdc1_al_hsv


#######################


def bmfl_al_extractor_small(bmfl_group, bmfl_count, config):
    # Source
    # Feature Extractor BMFL #01 for each fixture
    # |: 
    # 1 = Dim, 2 = ColorHue, 3 = ColorSat, 4 = Pan, 5 = Tilt,
    # 6 = Shutter, 7 = Zoom, 8 = Iris
    # :|
    # 
    # Target AL Extractor Small
    # 01 = Position of Max Intensity
    # 02 = Intensity of Peak absolute
    # 03 = Slope of Peak Intensity
    # 04 = AF Peak Density
    # 05 = Col Hue Mean
    # 06 = Col Sat Mean
    # 07 = Pan Mean
    # 08 = Tilt Mean
    #
    one_bmfl_extract_size = config.one_bmfl_extract_size
    len = bmfl_group.shape[0]
    bmfl_al_small = np.zeros((len, 8))

    dim_bmfl = param_extractor(bmfl_group, bmfl_count, one_bmfl_extract_size, 0)
    colhue_bmfl = param_extractor(bmfl_group, bmfl_count, one_bmfl_extract_size, 1)
    colsat_bmfl = param_extractor(bmfl_group, bmfl_count, one_bmfl_extract_size, 2)
    pan_bmfl = param_extractor(bmfl_group, bmfl_count, one_bmfl_extract_size, 3)
    tilt_bmfl = param_extractor(bmfl_group, bmfl_count, one_bmfl_extract_size, 4)

    maxvel_dim_bmfl, maxpos_dim_bmfl = find_maxvel_maxpos(dim_bmfl)
    bmfl_al_small[:, 0] = maxpos_dim_bmfl
    bmfl_al_small[:, 1] = maxvel_dim_bmfl
    bmfl_al_small[:, 2] = get_gradient(dim_bmfl)
    alternating_factors_dim_bmfl = get_alternating_factors(dim_bmfl)
    bmfl_al_small[:, 3] = alternating_factors_dim_bmfl[:, 0]
    bmfl_al_small[:, 4] = get_mean_vel(colhue_bmfl)
    bmfl_al_small[:, 5] = get_mean_vel(colsat_bmfl)
    bmfl_al_small[:, 6] = get_mean_vel(pan_bmfl)
    bmfl_al_small[:, 7] = get_mean_vel(tilt_bmfl)

    return bmfl_al_small


#######################


def jdc1_al_extractor_small(jdc1_group, jdc1_count, config):
    # Source
    # Feature Extractor JDC1 #01 for each fixture
    # |
    # 1 = Plate_Int, 2, Beam_Int,
    # 3 = ColorHue, 4 = ColorSat, 5 = ColorVel,
    # 6 = Tilt, 7 = Plate_Shutter, 8 = Plate_Rate,
    # 9 = Beam_Shutter, 10 = Beam_Rate
    # :|
    #
    # AL Extractor Small
    # 01 = Plate Position of Max Intensity
    # 02 = Plate Intensity of Peak absolute
    # 03 = Plate Slope of Peak Intensity
    # 04 = Plate AF Peak Density
    # 05 = Beam Intensity Mean
    # 06 = Col Hue
    # 07 = Col Sat
    # 08 = Tilt Mean
    #
    one_jdc1_extract_size = config.one_jdc1_extract_size
    len = jdc1_group.shape[0]
    jdc1_al_small = np.zeros((len, 8))

    int_jdc1_plate = param_extractor(jdc1_group, jdc1_count, one_jdc1_extract_size, 0)
    int_jdc1_beam = param_extractor(jdc1_group, jdc1_count, one_jdc1_extract_size, 1)
    colhue_jdc1_plate = param_extractor(jdc1_group, jdc1_count, one_jdc1_extract_size, 2)
    colsat_jdc1_plate = param_extractor(jdc1_group, jdc1_count, one_jdc1_extract_size, 3)
    tilt_jdc1 = param_extractor(jdc1_group, jdc1_count, one_jdc1_extract_size, 5)

    maxvel_int_jdc1, maxpos_int_jdc1 = find_maxvel_maxpos(int_jdc1_plate)
    jdc1_al_small[:, 0] = maxpos_int_jdc1
    jdc1_al_small[:, 1] = maxvel_int_jdc1
    jdc1_al_small[:, 2] = get_gradient(int_jdc1_plate)
    alternating_factors_int_jdc1 = get_alternating_factors(int_jdc1_plate)
    jdc1_al_small[:, 3] = alternating_factors_int_jdc1[:, 0]
    jdc1_al_small[:, 4] = get_mean_vel(int_jdc1_beam)
    jdc1_al_small[:, 5] = get_mean_vel(colhue_jdc1_plate)
    jdc1_al_small[:, 6] = get_mean_vel(colsat_jdc1_plate)
    jdc1_al_small[:, 7] = get_mean_vel(tilt_jdc1)

    return jdc1_al_small


#######################


def bmfl_al_extractor_large(bmfl_group, bmfl_count, config):
    # Source
    # Feature Extractor BMFL #01 for each fixture
    # |: 
    # 1 = Dim, 2 = ColorHue, 3 = ColorSat, 4 = Pan, 5 = Tilt,
    # 6 = Shutter, 7 = Zoom, 8 = Iris
    # :|
    # 
    # Target AL Extractor Large
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
    one_bmfl_extract_size = config.one_bmfl_extract_size
    len = bmfl_group.shape[0]
    bmfl_al_large = np.zeros((len, 18))
    #
    dim_bmfl = param_extractor(bmfl_group, bmfl_count, one_bmfl_extract_size, 0)
    colhue_bmfl = param_extractor(bmfl_group, bmfl_count, one_bmfl_extract_size, 1)
    colsat_bmfl = param_extractor(bmfl_group, bmfl_count, one_bmfl_extract_size, 2)
    pan_bmfl = param_extractor(bmfl_group, bmfl_count, one_bmfl_extract_size, 3)
    tilt_bmfl = param_extractor(bmfl_group, bmfl_count, one_bmfl_extract_size, 4)
    #
    ####
    ####
    maxvel_dim_bmfl, maxpos_dim_bmfl = find_maxvel_maxpos(dim_bmfl)
    bmfl_al_large[:, 0] = maxpos_dim_bmfl
    bmfl_al_large[:, 1] = maxvel_dim_bmfl
    bmfl_al_large[:, 2] = get_gradient(dim_bmfl)
    bmfl_al_large[:, 3:8] = get_alternating_factors(dim_bmfl)
    bmfl_al_large[:, 8], bmfl_al_large[:, 9], bmfl_al_large[:, 10], bmfl_al_large[:, 11] = \
        get_colhue_colsat_at_pos(colhue_bmfl, colsat_bmfl, dim_bmfl)
    bmfl_al_large[:, 12], bmfl_al_large[:, 13], bmfl_al_large[:, 14], bmfl_al_large[:, 15] = \
        get_pan_values_large(pan_bmfl)
    bmfl_al_large[:, 16], bmfl_al_large[:, 17] = get_tilt_values_large(tilt_bmfl)

    return bmfl_al_large


#######################


def get_mean_vel(group):
    len_group = group.shape[0]
    meanvel = np.zeros(len_group)

    for i in range(len_group):
        meanvel[i] = np.mean(group[i, :])

    return meanvel


#######################


def get_tilt_values_large(tilt_group):
    len = tilt_group.shape[0]
    tilt_max_val = np.zeros(len)
    tilt_min_val = np.zeros(len)
    maxpos = np.zeros(len, dtype=int)
    minpos = np.zeros(len, dtype=int)

    # Maxima absolute
    for i in range(len):
        maxpos[i] = np.argmax(tilt_group[i])

    # Minima absolute
    for i in range(len):
        minpos[i] = np.argmin(tilt_group[i])

    # Get the values
    for i in range(len):
        tilt_min_val[i] = tilt_group[i, minpos[i]]
        tilt_max_val[i] = tilt_group[i, maxpos[i]]

    return tilt_max_val, tilt_min_val


#######################


def get_pan_values_large(pan_group):
    len = pan_group.shape[0]
    width = pan_group.shape[1]

    pan_left_max_pos = np.zeros(len)
    pan_left_max_val = np.zeros(len)
    pan_right_max_pos = np.zeros(len)
    pan_right_max_val = np.zeros(len)

    maxvel = np.zeros((len, 1))
    maxpos = np.zeros((len, 1))
    maxposrel = np.zeros((len, 1))

    # Maxima absolute
    for i in range(len):
        maxvel[i], maxpos[i] = np.max(pan_group[i, :]), np.argmax(pan_group[i, :])
        maxposrel[i] = maxpos[i] / width

    minvel = np.zeros((len, 1))
    minpos = np.zeros((len, 1))
    minposrel = np.zeros((len, 1))

    # Minima absolute
    for i in range(len):
        minvel[i], minpos[i] = np.min(pan_group[i, :]), np.argmin(pan_group[i, :])
        minposrel[i] = minpos[i] / width


    # Get the values
    for i in range(len):
        # change this for len > 1
        maxpos = int(maxpos)
        minpos = int(minpos)
        #
        pan_left_max_pos[i] = minposrel[i]
        pan_left_max_val[i] = pan_group[i, minpos]
        pan_right_max_pos[i] = maxposrel[i]
        pan_right_max_val[i] = pan_group[i, maxpos]

    return pan_left_max_pos, pan_left_max_val, pan_right_max_pos, pan_right_max_val


#######################


def get_colhue_colsat_at_pos(colhue_group, colsat_group, dim):
    len = colhue_group.shape[0]

    colhue_lead = np.zeros(len)
    colsat_lead = np.zeros(len)
    colhue_back = np.zeros(len)
    colsat_back = np.zeros(len)

    maxvel = np.zeros((len, 1))
    maxpos = np.zeros((len, 1))

    # Maxima absolute
    for i in range(len):
        maxvel[i], maxpos[i] = np.max(dim[i, :]), np.argmax(dim[i, :])

    minvel = np.zeros((len, 1))
    minpos = np.zeros((len, 1))

    # Minima absolute
    for i in range(len):
        minvel[i], minpos[i] = np.min(dim[i, :]), np.argmin(dim[i, :])

    # get color max from brightest fixture in the group and vice versa for the darkest
    for i in range(len):
        # change this for len > 1
        maxpos = int(maxpos)
        minpos = int(minpos)
        #
        colhue_lead[i] = colhue_group[i, maxpos]
        colsat_lead[i] = colsat_group[i, maxpos]
        colhue_back[i] = colhue_group[i, minpos]
        colsat_back[i] = colsat_group[i, minpos]

    return colhue_lead, colsat_lead, colhue_back, colsat_back


#######################


def get_alternating_factors(group):
    len, width = group.shape

    # Feature Extractor Alternating Factors
    # 1 = density_peaks, 2 = peak_simularity,
    # 3 = Pos 2nd Peak (if exists), 4 = Intensity of Minima absolute,
    # 5 = Minima Simularity
    density_peaks = np.zeros(len)
    peak_simularity = np.zeros(len)
    minima_simularity = np.zeros(len)
    alternating_factors = np.zeros((len, 5))

    minvel, _ = find_minvel_minpos(group)
    # gradients = get_gradient(group)
    # print(group.shape)

    for i in range(len):
        # print(group)
        val_peaks, pos_peaks = find_peaks_np(group[i, :])
        # print(pos_peaks)
        peaks_len = val_peaks.size
        # print(peaks_len)
        size_pos_peaks = pos_peaks.size
        # print(val_peaks)
        density_peaks[i] = size_pos_peaks / width
        # print(density_peaks[i])
        # for the case that all values are the same the density will be 1. CHECK!?!?
        if peaks_len == 0:
            density_peaks[i] = 1
        alternating_factors[i, 0] = density_peaks[i]

        if peaks_len == 0:
            peak_simularity[i] = 1
        else:
            max_peak = np.max(val_peaks)
            min_peak = np.min(val_peaks)
            peak_simularity[i] = 1 - (max_peak - min_peak)

        alternating_factors[i, 1] = peak_simularity[i]
        # print(peak_simularity)

        if pos_peaks.size > 1:
            relative_pos_2nd_peak = pos_peaks[1] / width
            alternating_factors[i, 2] = relative_pos_2nd_peak
        elif pos_peaks.size == 0:
            alternating_factors[i, 2] = 0
        else:
            alternating_factors[i, 2] = pos_peaks[0] / width
        # print(alternating_factors[i, 2])

        alternating_factors[i, 3] = minvel[i]
        inverted_group = 1 - group[i, :]
        val_minima, _ = find_peaks_np(inverted_group)
        minima_len = val_minima.size

        if minima_len == 0:
            minima_simularity[i] = 1
        else:
            max_minima = np.max(val_minima)
            min_minima = np.min(val_minima)
            minima_simularity[i] = 1 - (max_minima - min_minima)

        alternating_factors[i, 4] = minima_simularity[i]
        # print(alternating_factors[i, 4])
        # print(alternating_factors)

    return alternating_factors


#######################


def find_maxvel_maxpos(group):
    len, width = group.shape
    maxvel = np.zeros((len, 1))
    maxpos = np.zeros((len, 1))

    # Maxima relative
    for i in range(len):
        maxvel[i], maxpos[i] = np.max(group[i, :]), np.argmax(group[i, :])
        maxpos[i] = maxpos[i] / width

    return maxvel, maxpos


#######################


def find_minvel_minpos(group):
    len, width = group.shape
    minvel = np.zeros((len, 1))
    minpos = np.zeros((len, 1))

    # Minima relative
    for i in range(len):
        minvel[i], minpos[i] = np.min(group[i, :]), np.argmin(group[i, :])
        minpos[i] = minpos[i] / width

    return minvel, minpos


#######################


def get_gradient(group):
    len, width = group.shape
    gradients = np.zeros((len, 1))

    maxvel = np.zeros((len, 1))
    maxpos = np.zeros((len, 1))

    # Maxima absolute
    for i in range(len):
        maxvel[i], maxpos[i] = np.max(group[i, :]), np.argmax(group[i, :])

    minvel = np.zeros((len, 1))
    minpos = np.zeros((len, 1))

    # Minima absolute
    for i in range(len):
        minvel[i], minpos[i] = np.min(group[i, :]), np.argmin(group[i, :])

    for i in range(len):
        gradient_segment = np.zeros((1, width + 4))
        gradient_segment[0, 2:width + 2] = group[i, :]
        mean_temp = np.gradient(gradient_segment, axis=1)
        mean_temp = np.abs(mean_temp)
        nonzero_indices = np.nonzero(mean_temp)
        mean_temp = mean_temp[nonzero_indices]

        mean_temp_calc = np.zeros((1, mean_temp.size))
        mean_temp_calc = mean_temp
        if mean_temp.size == 0:
            mean_temp_calc = np.zeros((1,1))

        # CHECK the *2 factor in the visualizer ?!?!?
        gradients[i] = bound((np.mean(mean_temp_calc) * 2), 0, 1)

    return gradients


#######################


def param_extractor(extract_group, count, one_fixture_size, param_pos):
    len = extract_group.shape[0]
    param_array = np.zeros((len, count))

    for i in range(count):
        k = i
        k = k * one_fixture_size + param_pos
        param_array[:, i] = extract_group[:, k]

    return param_array


#######################


def jdc1_extract_from_artnet(artnet_rec_v01, count, startaddress, config):
    
    artnet_rec_v01 = np.array(artnet_rec_v01)
    len_ = artnet_rec_v01.shape[0]

    one_jdc1_size = config.one_jdc1_size
    one_jdc1_extract_size = config.one_jdc1_extract_size

    jdc1_extract_array = np.zeros((len_, one_jdc1_extract_size * count))
    jdc1_region_artnet = artnet_rec_v01[:, startaddress:(startaddress + count * one_jdc1_size)]

    # Feature Extractor JDC1 #01
    # |: 
    # 1 = Plate_Int, 2, Beam_Int, 
    # 3 = ColorHue, 4 = ColorSat, 5 = ColorVel,
    # 6 = Tilt, 7 = Plate_Shutter, 8 = Plate_Rate, 
    # 9 = Beam_Shutter, 10 = Beam_Rate
    # :|

    for i in range(count):
        # counter Plate_Int source
        k = i
        k = k * one_jdc1_size - 55
        # counter Plate_Int target
        j = i * 10
        j = j - 10
        # set to extractor array Plate_Int
        plate_int = jdc1_region_artnet[:, k]
        jdc1_extract_array[:, j] = plate_int
        # print(plate_int)
        ####
        ####
        # counter Beam_Int source
        k = i
        k = k * one_jdc1_size - 60
        # counter Beam_Int target
        j = i * 10
        j = j - 9
        # set to extractor array Beam_Int
        beam_int = jdc1_region_artnet[:, k]
        jdc1_extract_array[:, j] = beam_int
        ####
        ####
        # Color
        # counter Plate first Pixel source
        k = i
        k_red = k * one_jdc1_size - 48
        k_green = k * one_jdc1_size - 47
        k_blue = k * one_jdc1_size - 46
        # Get RGB Colors
        h_temp = np.zeros((len_, 12))
        s_temp = np.zeros((len_, 12))
        v_temp = np.zeros((len_, 12))
        for n in range(12):
            v = n * 3
            h = v - 3
            s = v - 2
            rr = jdc1_region_artnet[:, (k_red + v)]
            gg = jdc1_region_artnet[:, (k_green + v)]
            bb = jdc1_region_artnet[:, (k_blue + v)]
            hh, ss, vv = colorsys.rgb_to_hsv(rr, gg, bb)
            h_temp[:, n] = hh
            s_temp[:, n] = ss
            v_temp[:, n] = vv
        h = np.mean(h_temp, axis=1)
        s = np.mean(s_temp, axis=1)
        v = np.mean(v_temp, axis=1)
        # counter color target
        j = i * 10
        jh = j - 8
        js = j - 7
        jv = j - 6
        # set to extractor array HSV
        jdc1_extract_array[:, jh] = h
        jdc1_extract_array[:, js] = s
        # jdc1_extract_array[:, jv] = v
        # maybe switch back to behaviour vel = dim
        jdc1_extract_array[:, jv] = plate_int
        ####
        ####
        # counter TILT source
        k = i
        k_msb = k * one_jdc1_size - 62
        k_lsb = k * one_jdc1_size - 61
        # counter TILT target
        j = i * 10
        j = j - 5
        # MSB combine LSB
        MSB = jdc1_region_artnet[:, k_msb]
        LSB = jdc1_region_artnet[:, k_lsb]
        jdc1_tilt = (MSB * 65536 + LSB * 256) / 65536
        # set to extractor array TILT
        jdc1_extract_array[:, j] = jdc1_tilt
        ####
        ####
        # counter Plate_Shutter source
        k = i
        k = k * one_jdc1_size - 52
        # counter Plate_Shutter target
        j = i * 10
        j = j - 4
        # set to extractor array Plate_Shutter
        plate_shutter = jdc1_region_artnet[:, k]
        jdc1_extract_array[:, j] = plate_shutter
        ####
        ####
        # counter Plate_Rate source
        k = i
        k = k * one_jdc1_size - 53
        # counter Plate_Rate target
        j = i * 10
        j = j - 3
        # set to extractor array Plate_Rate
        plate_rate = jdc1_region_artnet[:, k]
        jdc1_extract_array[:, j] = plate_rate
        ####
        ####
        # counter Beam_Shutter source
        k = i
        k = k * one_jdc1_size - 57
        # counter Beam_Shutter target
        j = i * 10
        j = j - 2
        # set to extractor array Beam_Shutter
        beam_shutter = jdc1_region_artnet[:, k]
        jdc1_extract_array[:, j] = beam_shutter
        ####
        ####
        # counter Beam_Rate source
        k = i
        k = k * one_jdc1_size - 58
        # counter Beam_Rate target
        j = i * 10
        j = j - 1
        # set to extractor array Beam_Rate
        beam_rate = jdc1_region_artnet[:, k]
        jdc1_extract_array[:, j] = beam_rate

    return jdc1_extract_array


#######################


def bmfl_extract_from_artnet(artnet_rec_v01, count, startaddress, config):

    artnet_rec_v01 = np.array(artnet_rec_v01)
    len_ = artnet_rec_v01.shape[0]
    one_bmfl_size = config.one_bmfl_size
    one_bmfl_extract_size = config.one_bmfl_extract_size

    bmfl_extract_array = np.zeros((len_, one_bmfl_extract_size * count))
    bmfl_region_artnet = artnet_rec_v01[:, startaddress:(startaddress + count * one_bmfl_size)]

    # Feature Extractor BMFL #01
    # |: 
    # 1 = Dim, 2 = ColorHue, 3 = ColorSat, 4 = Pan, 5 = Tilt,
    # 6 = Shutter, 7 = Zoom, 8 = Iris
    # :|
    #

    for i in range(count):
        # counter DIM source
        k = i
        k_msb = k * one_bmfl_size - 2
        k_lsb = k * one_bmfl_size - 1
        # MSB combine LSB
        MSB = bmfl_region_artnet[:, k_msb]
        LSB = bmfl_region_artnet[:, k_lsb]
        bmfl_dim = (MSB * 65536 + LSB * 256) / 65536
        ####
        ####
        # counter color source
        k = i
        k_cyan = k * one_bmfl_size - 38
        k_mag = k * one_bmfl_size - 37
        k_yell = k * one_bmfl_size - 36
        k_cto = k * one_bmfl_size - 35

        cyan = bmfl_region_artnet[:, k_cyan]
        mag = bmfl_region_artnet[:, k_mag]
        yell = bmfl_region_artnet[:, k_yell]
        cto = bmfl_region_artnet[:, k_cto]

        red = 1 - cyan
        green = 1 - mag
        blue = 1 - yell
        # shift for cto ## overhaul ##
        blue = blue - 0.25 * cto
        blue = np.vectorize(bound)(blue, 0, 1)
        green = green - 0.1 * cto
        green = np.vectorize(bound)(green, 0, 1)
        # counter color target
        j = i * 8
        n = j - 7
        l = j - 6
        # color temp store
        color_temp_rgb = np.zeros((len_, 3))
        color_temp_hsv = np.zeros((len_, 3))
        # extract red, green & blue
        color_temp_rgb[:, 0] = red
        color_temp_rgb[:, 1] = green
        color_temp_rgb[:, 2] = blue
        # convert rgb to hsv
        for r, g, b in color_temp_rgb:
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            color_temp_hsv[:, 0] = h
            color_temp_hsv[:, 1] = s
        # set to extractor array COLOR_HUE and COLOR_SAT
        bmfl_extract_array[:, n] = color_temp_hsv[:, 0]
        bmfl_extract_array[:, l] = color_temp_hsv[:, 1]
        ####
        ####
        # counter PAN source
        k = i
        k_msb = k * one_bmfl_size - 48
        k_lsb = k * one_bmfl_size - 47
        # counter PAN target
        j = i * 8
        j = j - 5
        # MSB combine LSB
        MSB = bmfl_region_artnet[:, k_msb]
        LSB = bmfl_region_artnet[:, k_lsb]
        bmfl_pan = (MSB * 65536 + LSB * 256) / 65536
        # set to extractor array PAN
        bmfl_extract_array[:, j] = bmfl_pan
        ####
        ####
        # counter TILT source
        k = i
        k_msb = k * one_bmfl_size - 46
        k_lsb = k * one_bmfl_size - 45
        # counter TILT target
        j = i * 8
        j = j - 4
        # MSB combine LSB
        MSB = bmfl_region_artnet[:, k_msb]
        LSB = bmfl_region_artnet[:, k_lsb]
        bmfl_tilt = (MSB * 65536 + LSB * 256) / 65536
        # set to extractor array TILT
        bmfl_extract_array[:, j] = bmfl_tilt
        ####
        ####
        # counter Iris source
        k = i
        k_msb = k * one_bmfl_size - 21
        k_lsb = k * one_bmfl_size - 20
        # counter Iris target
        j = i * 8
        j = j - 1
        # MSB combine LSB
        MSB = bmfl_region_artnet[:, k_msb]
        LSB = bmfl_region_artnet[:, k_lsb]
        bmfl_iris = (MSB * 65536 + LSB * 256) / 65536
        # set to extractor array Iris
        bmfl_extract_array[:, j] = bmfl_iris
        ####
        ####
        # counter Zoom source
        k = i
        k_msb = k * one_bmfl_size - 19
        k_lsb = k * one_bmfl_size - 18
        # counter Zoom target
        j = i * 8
        j = j - 2
        # MSB combine LSB
        MSB = bmfl_region_artnet[:, k_msb]
        LSB = bmfl_region_artnet[:, k_lsb]
        bmfl_zoom = (MSB * 65536 + LSB * 256) / 65536
        # set to extractor array Zoom
        bmfl_extract_array[:, j] = bmfl_zoom
        ####
        ####
        # counter Shutter source
        k = i
        k = k * one_bmfl_size - 3
        # counter Shutter target
        j = i * 8
        j = j - 3
        # counter DIM target
        d = i * 8
        d = d - 8
        # set to extractor array Shutter
        bmfl_extract_array[:, j] = bmfl_region_artnet[:, k]
        bmfl_shutter = bmfl_region_artnet[:, k]
        # shutter adoption v01
        ####
        #### 
        # to be overhauled for the creating of non random dimmer values
        for n in range(len_):
            if (bmfl_shutter[n] >= (32/256)) and (bmfl_shutter[n] <= (63/256)):
                # set to extractor array DIM
                bmfl_extract_array[n, d] = bmfl_dim[n]
            elif (bmfl_shutter[n] >= (64/256)) and (bmfl_shutter[n] <= (95/256)):
                ####
                # adopt for strobe ranges in 64-95 and 192-223
                ####
                randomizer = np.random.rand()
                if randomizer > 0.5:
                    bmfl_extract_array[n, d] = bmfl_dim[n]
                else:
                    bmfl_extract_array[n, d] = 0.0
            elif (bmfl_shutter[n] >= (96/256)) and (bmfl_shutter[n] <= (127/256)):
                bmfl_extract_array[n, d] = bmfl_dim[n]
            elif (bmfl_shutter[n] >= (192/256)) and (bmfl_shutter[n] <= (223/256)):
                ####
                # adopt for random strobe ranges in 64-95 and 192-223
                ####
                randomizer = np.random.rand()
                if randomizer > 0.5:
                    bmfl_extract_array[n, d] = bmfl_dim[n]
                else:
                    bmfl_extract_array[n, d] = 0.0
            elif (bmfl_shutter[n] >= (224/256)) and (bmfl_shutter[n] <= (255/256)):
                bmfl_extract_array[n, d] = bmfl_dim[n]
            else:
                bmfl_extract_array[n, d] = 0.0

    return bmfl_extract_array


#######################


def find_peaks_np(array):
    peak_indices = []
    peak_values = []
    same_values = (array[0] == array).all()

    # If the entire array has the same values, return empty lists
    if same_values:
        return np.array(peak_values), np.array(peak_indices)

    # Check if the first element is the highest
    if array[0] > array[1]:
        # print('first')
        peak_indices.append(0)
        peak_values.append(array[0])

    # Slide along the array and check if the current element is greater than to its neighbors
    for i in range(1, len(array) - 1):
        if array[i] > array[i - 1] and array[i] > array[i + 1]:
            # print('mid')
            # print(i)
            peak_indices.append(i)
            peak_values.append(array[i])

    """
    # Slide along the array and check if the current peak element is equal to its neighbors
    if len(peak_indices) != 0:
        for i in range(peak_indices[0], (peak_indices[-1] - peak_indices[0] + 1)):
            # print(i)
            #if array[i] == array[i - 1] and array[i] == array[i + 1]:
            if array[i] == array[i - 1] and array[peak_indices[0]] == array[i]:
                # print(i)
                peak_indices.append(i)
                peak_values.append(array[i])
    """

    # Check if the last element is the highest
    if array[-1] > array[-2]:
        # print('last')
        peak_indices.append(len(array) - 1)
        peak_values.append(array[-1])

    # print(peak_indices)

    return np.array(peak_values), np.array(peak_indices)


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

    with open('artnet_file.json', 'r') as f:
        data = json.load(f)
	artnet_dummy_temp =  = np.array(data)

    artnet_dummy = np.zeros((artnet_dummy_temp.shape[0], config.len_16uni))

    for i in range (artnet_dummy_temp.shape[0]):
        for j in range (config.len_16uni) :
            artnet_dummy[i,j] = artnet_dummy_temp[i,j] / 256
        ### Main Loop
        features_72params = extract_features_Artnet_to_72params_v02(artnet_dummy[i,:],config)
        ###

    # Write JSON file to disk
    #
    # Convert the numpy array to a list
    features_72params_list = features_72params.tolist()
    with open('al_file.json', 'w') as f:
        json.dump(array_list, f)

