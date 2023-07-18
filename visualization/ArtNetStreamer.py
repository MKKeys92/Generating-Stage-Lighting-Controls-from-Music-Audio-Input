import json
import threading
import math
import time
import numpy as np
from playsound import playsound
from stupidArtnet import StupidArtnet
import time
import random

# THESE ARE MOST LIKELY THE VALUES YOU WILL BE NEEDING
target_ip = '192.168.178.93'		# typically in 2.x or 10.x range
universe = 0 										# see docs
packet_size = 512								# it is not necessary to send whole universe

global stop_artnet
dmx_frame_ns = 33333333
dmx_delay_ms = 4/30 * 1000
audio_delay_ms = 0
test_data = list(np.ones(254) * 255)

artnet_json = "/home/michaelk/PycharmProjects/alternator_v1.1/predict/output/artnet_1024_ID_Test05_ConcreteSchoolyard.json"
audio_path = "/home/ma/PycharmProjects/alternator_v1.1/preprocessing/raw_data/Deichkind/audio/DK2020-18-LeiderGeil_02.wav"
numpy_path = "/home/ma/PycharmProjects/alternator_v1.1/preprocessing/datasets/artnet_DK2020_18_LDGL02.npz"
input_type = "numpy"

def send_artnet_frame(number, artnet_nodes, data):
    print('sending frame ' + str(number))

    for i in range(len(artnet_nodes)):
        artnet_nodes[i].set(data[i][number])


def play_audio():
    if audio_delay_ms != 0:
        delay_start = time.perf_counter_ns()
        delay_end = delay_start + audio_delay_ms*1000000
        while True:
            if time.perf_counter_ns() > delay_end:
                break

    playsound(audio_path)

def artnet_thread(name, artnet_nodes, data):
    if dmx_delay_ms != 0:
        delay_start = time.perf_counter_ns()
        delay_end = delay_start + dmx_delay_ms*1000000
        while True:
            if stop_artnet:
                break
            if time.perf_counter_ns() > delay_end:
                break
    min_n = len(data[0])
    for d in data:
        min_n = min(min_n, len(d))

    start_time = time.perf_counter_ns()
    send_artnet_frame(0, artnet_nodes, data)
    last_frame = 0
    while True:
        if stop_artnet or last_frame == min_n-1:
            break
        n = math.floor(((time.perf_counter_ns() - start_time) / 1000000000) * 30)
        if n > last_frame:
            send_artnet_frame(n, artnet_nodes, data)
            last_frame = n

def np_to_packet(arr):
    d = []

    for r in arr:
        packet = bytearray(len(r))
        for i in range(len(r)):
            packet[i] = int(r[i] * 255)
        d.append(packet)
    return d

if __name__ == "__main__":

    # CREATING A STUPID ARTNET OBJECT
    # SETUP NEEDS A FEW ELEMENTS
    # TARGET_IP   = DEFAULT 127.0.0.1
    # UNIVERSE    = DEFAULT 0
    # PACKET_SIZE = DEFAULT 512
    # FRAME_RATE  = DEFAULT 30
    # ISBROADCAST = DEFAULT FALSE
    nr_universe = 16
    artnet_sender = []
    artnet_data = []

    if input_type == "json" :
        with open(artnet_json) as f:
            dict = json.loads(f.read())
            for k,v in dict.items():
                if(k.startswith('lighting_array')):
                    arr = np.array(v)
                    d = np_to_packet(arr)
                    artnet_data.append(d)
    elif input_type == "numpy":
        arr = np.load(numpy_path, allow_pickle=True)['lighting_array']
        arr = np.reshape(arr, (-1, 16, 512))
        for k in range(16):
            a = arr[:,k,:]
            d = np_to_packet(a)
            artnet_data.append(d)
    else:
        assert('input type is not available')

    for i in range(nr_universe):
        artnet_sender.append(StupidArtnet(target_ip, i, packet_size, 30, True, True))

    stop_artnet = False
    for s in artnet_sender:
        s.start()
    t1 = threading.Thread(target=artnet_thread, args=("ArtnetSender", artnet_sender, artnet_data))
    t1.start()
    play_audio()
    stop_artnet = True
    t1.join()

    # SOME DEVICES WOULD HOLD LAST DATA, TURN ALL OFF WHEN DONE

    for s in artnet_sender:
        s.blackout()

    # ... REMEMBER TO CLOSE THE THREAD ONCE YOU ARE DONE
    for s in artnet_sender:
        s.stop()

    # CLEANUP IN THE END
    for s in artnet_sender:
        del s

    print('thread killed')