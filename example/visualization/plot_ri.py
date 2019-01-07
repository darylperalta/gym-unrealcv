#!/usr/bin/env python

import os
import gym
import matplotlib
import matplotlib.pyplot as plt
import sys
import argparse
import json

def pause():
    programPause = input("Press the <ENTER> key to finish...")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--path", type=str, default='../ddpg_icm_sfm_combined_opt_rotreward_distTarget_enc/rewards.json', help="the path of monitor file")
    parser.add_argument("-s", "--save_interval", type=int, nargs='?',default = 25, help="save interval")
    # parser.add_argument("-f", "--full", action='store_true', help="print the full data plot with lines")
    # parser.add_argument("-d", "--dots", action='store_true', help="print the full data plot with dots")
    # parser.add_argument("-a", "--average", type=int, nargs='?', const=100, metavar="N", help="plot an averaged graph using N as average size delimiter. Default = 50")
    # parser.add_argument("-i", "--interpolated", type=int, nargs='?', const=50, metavar="M", help="plot an interpolated graph using M as interpolation amount. Default = 50")
    args = parser.parse_args()
    print(args.save_interval)
    #print args.path
    with open(args.path) as json_file:
        data = json.load(json_file)
        print(type(data['rewards_i']))
        print(list(range(0,len(data['rewards_i'])*args.save_interval,args.save_interval)))
    matplotlib.rcParams['toolbar'] = 'None'
    plt.style.use('ggplot')
    plt.xlabel("Episode")
    plt.ylabel("Cumulated Intrinsic Reward in an episode")
    fig = plt.gcf().canvas.set_window_title('averaged_simulation_graph')
    matplotlib.rcParams.update({'font.size': 15})
    plt.plot(list(range(0,len(data['rewards_i'])*args.save_interval,args.save_interval)),data['rewards_i'], color='red', linewidth=2.5)
    plt.pause(0.000001)
    pause()
