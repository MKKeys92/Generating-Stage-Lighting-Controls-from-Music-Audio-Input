from evaluator import *
import numpy as np
from scipy.signal import find_peaks
import torch


def main():
    beats = [   0,  0,  1,  0,  0,  1,  0,  0,  0,  1 , 0]
    val =   [   0.5,0.6,0.8,0.3,0.2,0.1,0.6,0.7,1.0,0.7,0.6 ]

    le = LightEvaluator(None,None,None)

    beats=np.array(beats).reshape((len(beats),1))
    val =np.array(val).reshape((len(beats),1))
    mask = np.ones((1,1))

    #test only makes sense with fitting prominence and no distance
    res = le.calc_beat_align_score(val,mask,mask,beats)
    ms = le.get_movement_score(torch.tensor(val), val, [2, 5])

    val = [0,0,0,1,1,1,1,0,0,0,1,1,1,0]
    peaks, properties = find_peaks(val)


    print('Done')


if __name__ == '__main__':
    main()