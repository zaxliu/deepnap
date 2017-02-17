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

exp_configs = [
    # (logprefix, g, phi, buf, rs, weight, start_time, backoff)
    # Fig 2
    ('message_2016-6-8_XXX',             0.99, 5, (1, 400), (1, 'adaptive'), None, '2014-09-25 09:20:00',  0),
    # Fig 3
    ('message_2016-6-11_1230_FR1000_G5',  0.5, 5, (2, 200), (1000, 'fixed'), None, '2014-09-25 09:20:00', None),
    ('message_2016-6-11_1230_FR20_G5',    0.5, 5, (2, 200), (20,   'fixed'), None, '2014-09-25 09:20:00', None),
    ('message_2016-6-11_1230_FR1_G5',     0.5, 5, (2, 200), (1,    'fixed'), None, '2014-09-25 09:20:00', None),
    # Fig 4
    # Fig 5
    ('message_2016-6-8_2130_AR1',         0.5, 5, (2, 200), (1, 'adaptive'), None, '2014-11-01 00:00:00', None),
    # Fig 6
    # Fig 7
    # Fig 8
    # Fig 9
    ('message_2016-6-13_G5_BUF1_FR20_1_1', 0.5, 5, (1, 400), (20,   'fixed'), 0.7, '2014-09-25 09:20:00', None),
    ('message_2016-6-13_G5_BUF1_FR20_1_2', 0.5, 5, (1, 400), (20,   'fixed'), 0.7, '2014-09-25 09:20:00', None),
    ('message_2016-6-13_G5_BUF2_FR20_1',   0.5, 5, (1, 400), (20,   'fixed'), 0.7, '2014-09-25 09:20:00', None),
    # Fig 10
    ('message_2016-6-12_G5_BUF2_AR1',     0.5, 5, (2, 200), (1, 'adaptive'), 0.5, '2014-10-15 09:20:00', None),
    # Fig 11
    ('message_2016-6-12_G5_BUF2_AR1_b1',  0.5, 5, (2, 200), (1, 'adaptive'), 0.1,  '2014-11-05 09:20:00', None),
    ('message_2016-6-12_G5_BUF2_AR1_b15', 0.5, 5, (2, 200), (1, 'adaptive'), 0.15, '2014-11-05 09:20:00', None),
    ('message_2016-6-12_G5_BUF2_AR1_b2',  0.5, 5, (2, 200), (1, 'adaptive'), 0.2,  '2014-11-05 09:20:00', None),
    ('message_2016-6-12_G5_BUF2_AR1_b25', 0.5, 5, (2, 200), (1, 'adaptive'), 0.25, '2014-11-05 09:20:00', None),
    ('message_2016-6-12_G5_BUF2_AR1_b3',  0.5, 5, (2, 200), (1, 'adaptive'), 0.3,  '2014-11-05 09:20:00', None),
    ('message_2016-6-12_G5_BUF2_AR1_b35', 0.5, 5, (2, 200), (1, 'adaptive'), 0.35, '2014-11-05 09:20:00', None),
    ('message_2016-6-12_G5_BUF2_AR1_b4',  0.5, 5, (2, 200), (1, 'adaptive'), 0.4,  '2014-11-05 09:20:00', None),
    ('message_2016-6-12_G5_BUF2_AR1_b5',  0.5, 5, (2, 200), (1, 'adaptive'), 0.5,  '2014-11-05 09:20:00', None),
    ('message_2016-6-12_G5_BUF2_AR1_b55', 0.5, 5, (2, 200), (1, 'adaptive'), 0.5, '2014-10-15 09:40:00', None),
    ('message_2016-6-12_G5_BUF2_AR1_b6',  0.5, 5, (2, 200), (1, 'adaptive'), 0.6,  '2014-11-05 09:20:00', None),
    ('message_2016-6-12_G5_BUF2_AR1_b65', 0.5, 5, (2, 200), (1, 'adaptive'), 0.65, '2014-11-05 09:20:00', None),
    ('message_2016-6-12_G5_BUF2_AR1_b7',  0.5, 5, (2, 200), (1, 'adaptive'), 0.7,  '2014-11-05 09:20:00', None),
    ('message_2016-6-12_G5_BUF2_AR1_b8',  0.5, 5, (2, 200), (1, 'adaptive'), 0.8,  '2014-11-05 09:20:00', None),
    # Fig 12
    ('message_2016-6-12_G9_BUF2_AR1',   0.9, 5, (2, 200), (1, 'adaptive'), 0.5, '2014-11-05 09:20:00', None),
    ('message_2016-6-12_G9_BUF2_FR100', 0.9, 5, (2, 200), (1, 'adaptive'), 0.5, '2014-11-05 09:20:00', None),
    ('message_2016-6-12_G9_BUF2_FR20',  0.9, 5, (2, 200), (1, 'adaptive'), 0.5, '2014-11-05 09:20:00', None),
    ('message_2016-6-12_G9_BUF2_FR1',   0.9, 5, (2, 200), (1, 'adaptive'), 0.5, '2014-11-05 09:20:00', None),
    ('message_2016-6-11_BUF2_G5',       0.5, 5, (2, 200), (1, 'adaptive'), 0.5, '2014-11-05 09:20:00', None),
    ('message_2016-6-11_BUF2_G5_FR100', 0.5, 5, (2, 200), (1, 'adaptive'), 0.5, '2014-11-05 09:20:00', None),
    ('message_2016-6-11_BUF2_G5_FR1',   0.5, 5, (2, 200), (1, 'adaptive'), 0.5, '2014-11-05 09:20:00', None),    
]

# create exp files
for logprefix, g, phi, buf, rs, weight, start_time, backoff in exp_configs:
    os.system('cp ./kdd-exps/experiment_QNN_legacy_template.py ./kdd-exps/experiment_{}_legacy.py'.format(logprefix))

# set params

for logprefix, g, phi, buf, rs, weight, start_time, backoff in exp_configs:
    with open("./kdd-exps/experiment_{}_legacy.py".format(logprefix), 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "log_file_name = " in line:
            lines[i] = "log_file_name = '{}.log'\n".format(logprefix)
        elif "gamma, alpha =" in line:
            lines[i] = "gamma, alpha = {}, 0.9\n".format(g)
        elif "num_buffer, memory_size = " in line:
            lines[i] = "num_buffer, memory_size = {}, {}\n".format(buf[0], buf[1])
        elif "reward_scaling, reward_scaling_update = " in line:
            lines[i] = "reward_scaling, reward_scaling_update = {}, '{}'\n".format(rs[0], rs[1])
        elif "beta = None" in line and weight is not None:
            lines[i] = "beta = {}\n".format(weight)
        elif "start_time = pd.to_datetime" in line:
            lines[i] = "start_time = pd.to_datetime('{}')\n".format(start_time)
        elif "backoff_epochs = num_buffer*memory_size+phi_length" in line and backoff is not None:
            lines[i] = "backoff_epochs = {}\n".format(backoff)
    print>>open("./kdd-exps/experiment_{}_legacy.py".format(logprefix), 'w'), ''.join(lines)