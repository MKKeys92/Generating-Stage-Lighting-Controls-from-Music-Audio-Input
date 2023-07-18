# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


""" This script handling the nn output to artnet test structure porcess. """
import os
import json
import argparse
import csv
# print(csv.__file__)
import numpy as np
# save numpy array as csv file
from numpy import asarray
from numpy import savetxt
from numpy import save
#import torch
#import torch.utils.data
#import matplotlib


parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='data',
                    help='the directory of csv data')

args = parser.parse_args()



def load_csv_lighting(csv_file):

    lighting_array = []

     ## read csv form hdd 
    with open(csv_file, 'r') as f:
        lighting_array = list(csv.reader(f, delimiter=","))
        lighting_array = np.array(lighting_array)

    #x = list(lighting_array.size())
    x = lighting_array.shape
    print(x)

    return lighting_array


def combine_arrays(lighting_from_nn,artnet_array):

    #test = lighting_from_nn[:,0]
    #print(test)
    #test2 = artnet_array[:,0]
    #print(test2)
    artnet_array_temp = artnet_array

    # rearrange to fit into alternator testsetup artnet stream to visualise
    artnet_array_temp[:,0] = lighting_from_nn[:,0]



    artnet_array = artnet_array_temp

    return artnet_array



def main():

    csv_file = ('data/nn_features_hypa_20220811.csv')
    lighting_from_nn = load_csv_lighting(csv_file)
    lighting_from_nn = lighting_from_nn.astype(float)

    csv_file = ('data/empty_artnet.csv')
    artnet_array = load_csv_lighting(csv_file)
    artnet_array = artnet_array.astype(float)
    artnet_array = artnet_array / 256

    combine_arrays(lighting_from_nn,artnet_array)
    

    


if __name__ == '__main__':
    main()
