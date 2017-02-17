import os
import sys
import time
from time import sleep
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Pool
import subprocess
sys.path.append('../')

# exp_configs = [
#     ('experiment_DynaQtable_template.py', 5, 1),
#     ('experiment_DynaQNN_template.py', 5, 1),
#     ('experiment_phiNN_template.py', None, 1)
# ]

exp_configs = [
    # (file, num_sim, n_bin, phi, location)
    # dh3
    ('experiment_DynaQtable_130_Feb12_2217.py', 5, 5, None, 'dh3'),  # n_sim=5, n_bin=5
    ('experiment_DynaQNN_Feb15_2000.py', 0, 0, None, 'dh3'),  # num_sim=0, n_bins=inf
    ('experiment_DynaQNN_130_Feb12_2215.py', 5, 0, None, 'dh3'),  # n_sim=5, n_bin=inf
    ('experiment_QNN_Jan31_1154.py', None, None, 15, 'dh3'),  # Phi=15
    # dsy
    ('experiment_DynaQtable_Feb12_2232.py', 5, 5, None, 'dsy'),  # n_sim=5, n_bin=5
    ('experiment_DynaQNN_Feb15_2050.py', 0, 0, None, 'dsy'),  # num_sim=0, n_bins=inf
    ('experiment_DynaQNN_Feb12_2226.py', 5, 0, None, 'dsy'),  # n_sim=5, n_bin=inf
    ('experiment_QNN_Feb1_1740.py', None, None, 15, 'dsy'),  # Phi=15
    # dmW
    ('experiment_DynaQtable_Feb7_1052.py', 5, 5, None, 'dmW'),  # n_sim=5, n_bin=5
    ('experiment_DynaQNN_130_Feb10_2316.py', 0, 0, None, 'dmW'),  # num_sim=0, n_bins=inf
    ('experiment_DynaQNN_Feb5_1007.py', 5, 0, None, 'dmW'),  # n_sim=5, n_bin=inf
    ('experiment_QNN_Feb2_0944.py', None, None, 15, 'dmW'),  # Phi=15
    # mhC
    ('experiment_DynaQtable_130_Feb14_0027.py', 5, 5, None, 'mhC'),  # n_sim=5, n_bin=5
    ('experiment_DynaQNN_130_Feb15_2001.py', 0, 0, None, 'mhC'),  # num_sim=0, n_bins=inf
    ('experiment_DynaQNN_130_Feb14_0026.py', 5, 0, None, 'mhC'),  # n_sim=5, n_bin=inf
    ('experiment_QNN_Feb2_0930.py', None, None, 15, 'mhC'),  # Phi=15
    # mdB
    ('experiment_DynaQtable_Feb13_2359.py', 5, 5, None, 'mdB'),  # n_sim=5, n_bin=5
    ('experiment_DynaQNN_Feb15_2051.py', 0, 0, None, 'mdB'),  # num_sim=0, n_bins=inf
    ('experiment_DynaQNN_Feb13_2358.py', 5, 0, None, 'mdB'),  # n_sim=5, n_bin=inf
    ('experiment_QNN_Feb2_0953.py', None, None, 15, 'mdB'),  # Phi=15
    # gym
    ('experiment_DynaQtable_130_Feb14_0029.py', 5, 5, None, 'gym'),  # n_sim=5, n_bin=5
    ('experiment_DynaQNN_130_Feb15_2002.py', 0, 0, None, 'gym'),  # num_sim=0, n_bins=inf
    ('experiment_DynaQNN_130_Feb14_0028.py', 5, 0, None, 'gym'),  # n_sim=5, n_bin=inf
    ('experiment_QNN_Feb2_1004.py', None, None, 15, 'gym'),  # Phi=15
    # dmW, different num state bins, n_sim=5
    ('experiment_DynaQtable_Feb7_1324.py', 5, 2, None, 'dmW'),  # n_bins=2
    # ('experiment_DynaQtable_Feb7_1052.py', 5, 5, None, 'dmW'),  # n_bins=5
    ('experiment_DynaQtable_Feb7_1609.py', 5, 7, None, 'dmW'),  # n_bins=7
    ('experiment_DynaQtable_Feb6_2008.py', 5, 10, None, 'dmW'),  # n_bins=10
    ('experiment_DynaQtable_Feb7_1053.py', 5, 15, None, 'dmW'),  # n_bins=15
    ('experiment_DynaQtable_Feb6_2010.py', 5, 25, None, 'dmW'),  # n_bins=25
    ('experiment_DynaQtable_Feb6_1543.py', 5, 50, None, 'dmW'),  # n_bins=50
    ('experiment_DynaQtable_Feb2_0946.py', 5, 100, None, 'dmW'),  # n_bins=100
    ('experiment_DynaQtable_Feb6_1544.py', 5, 250, None, 'dmW'),  # n_bins=250
    # dmW, different num simulated experiences
    # ('experiment_DynaQNN_130_Feb10_2316.py', 0, 0, None, 'dmW'),  # n_sim=0
    ('experiment_DynaQNN_130_Feb10_2317.py', 2, 0, None, 'dmW'),  # n_sim=2
    ('experiment_DynaQNN_Feb5_1007.py', 5, 0, None, 'dmW'),  # n_sim=5
    ('experiment_DynaQNN_Feb10_2300.py', 10, 0, None, 'dmW'),  # n_sim=10
    ('experiment_DynaQNN_Feb10_2305.py', 12, 0, None, 'dmW'),  # n_sim=12
    ('experiment_DynaQNN_Feb10_2302.py', 16, 0, None, 'dmW'),  # n_sim=16
    ('experiment_DynaQNN_Feb10_2303.py', 20, 0, None, 'dmW'),  # n_sim=20
]

# create exp files
for fname, _, _, _, _ in exp_configs:
    if 'DynaQNN' in fname:
        os.system('cp ./kdd-exps/experiment_DynaQNN_template.py ./kdd-exps/{}'.format(fname))
    elif 'DynaQtable' in fname:
        os.system('cp ./kdd-exps/experiment_DynaQtable_template.py ./kdd-exps/{}'.format(fname))
    elif '_QNN_' in fname:
        os.system('cp ./kdd-exps/experiment_phiNN_template.py ./kdd-exps/{}'.format(fname))
    else:
        print fname + ' invalid'

# set params
for fname, n_sim, n_bin, phi, loc in exp_configs:
    with open('./kdd-exps/'+fname, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "log_prefix = '_'.join(['msg'] + os.path.basename(__file__).split('_')" in line:
            lines[i] = "log_prefix = '_'.join(['msg'] + os.path.basename(__file__).replace('.', '_').split('_')[1:5])\n" if "_130_" in fname \
                    else "log_prefix = '_'.join(['msg'] + os.path.basename(__file__).replace('.', '_').split('_')[1:4])\n"
        if 'total_time = pd.Timedelta(' in line:
            lines[i] = 'total_time = pd.Timedelta(days=7)\n'
        elif 'num_sim = ' in line and n_sim is not None and 'Dyna' in fname:
            lines[i] = 'num_sim = {}\n'.format(n_sim)
        elif 'n_belief_bins, max_queue_len = ' in line and n_bin is not None and 'Dyna' in fname:
            lines[i] = 'n_belief_bins, max_queue_len = {}, 20\n'.format(n_bin)
        elif 'phi_length = ' in line and phi is not None and 'Dyna' not in fname:
            lines[i] = 'phi_length = {}\n'.format(phi)
        elif 'location = ' in line:
            lines[i] = "location = '{}'\n".format(loc)            
    print>>open('./kdd-exps/'+fname, 'w'), ''.join(lines)