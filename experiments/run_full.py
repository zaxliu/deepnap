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

#############################################################
# Configurations Below
#############################################################
previous_pid = None  # Unix PID of previous run scripts
n_process_exp = 2   # Num of parallel experiment runs
n_process_idx = 1   # Num of parallel log indexing jobs
# Task config tuples
exp_configs = [
    # (file, num_sim, num_log)
    # dh3
    ('experiment_DynaQtable_130_Feb12_2217.py', 5, 14),  # n_sim=5, n_bin=5
    ('experiment_DynaQNN_Feb15_2000.py', 0, 14),  # num_sim=0, n_bins=inf
    ('experiment_DynaQNN_130_Feb12_2215.py', 5, 14),  # n_sim=5, n_bin=inf
    ('experiment_QNN_Jan31_1154.py', None, 10),  # Phi=15
    # dsy
    ('experiment_DynaQtable_Feb12_2232.py', 5, 14),  # n_sim=5, n_bin=5
    ('experiment_DynaQNN_Feb15_2050.py', 0, 14),  # num_sim=0, n_bins=inf
    ('experiment_DynaQNN_Feb12_2226.py', 5, 14),  # n_sim=5, n_bin=inf
    ('experiment_QNN_Feb1_1740.py', None, 14),  # Phi=15
    # dmW
    ('experiment_DynaQtable_Feb7_1052.py', 5, 14),  # n_sim=5, n_bin=5
    ('experiment_DynaQNN_130_Feb10_2316.py', 0, 14),  # num_sim=0, n_bins=inf
    ('experiment_DynaQNN_Feb5_1007.py', 5, 14),  # n_sim=5, n_bin=inf
    ('experiment_QNN_Feb2_0944.py', None, 14),  # Phi=15
    # mhC
    # ('experiment_DynaQtable_130_Feb14_0027.py', 5, 14),  # n_sim=5, n_bin=5
    ('experiment_DynaQNN_130_Feb15_2001.py', 0, 14),  # num_sim=0, n_bins=inf
    ('experiment_DynaQNN_130_Feb14_0026.py', 5, 14),  # n_sim=5, n_bin=inf
    ('experiment_QNN_Feb2_0930.py', None, 14),  # Phi=15
    # mdB
    ('experiment_DynaQtable_Feb13_2359.py', 5, 14),  # n_sim=5, n_bin=5
    ('experiment_DynaQNN_Feb15_2051.py', 0, 14),  # num_sim=0, n_bins=inf
    ('experiment_DynaQNN_Feb13_2358.py', 5, 14),  # n_sim=5, n_bin=inf
    ('experiment_QNN_Feb2_0953.py', None, 14),  # Phi=15
    # gym
    ('experiment_DynaQtable_130_Feb14_0029.py', 5, 14),  # n_sim=5, n_bin=5
    ('experiment_DynaQNN_130_Feb15_2002.py', 0, 14),  # num_sim=0, n_bins=inf
    ('experiment_DynaQNN_130_Feb14_0028.py', 5, 14),  # n_sim=5, n_bin=inf
    ('experiment_QNN_Feb2_1004.py', None, 14),  # Phi=15
    # dmW, different num state bins, n_sim=5
    ('experiment_DynaQtable_Feb7_1324.py', 5, 14),  # n_bins=2
    ('experiment_DynaQtable_Feb7_1052.py', 5, 14),  # n_bins=5
    ('experiment_DynaQtable_Feb7_1609.py', 5, 14),  # n_bins=7
    ('experiment_DynaQtable_Feb6_2008.py', 5, 14),  # n_bins=10
    ('experiment_DynaQtable_Feb7_1053.py', 5, 14),  # n_bins=15
    ('experiment_DynaQtable_Feb6_2010.py', 5, 14),  # n_bins=25
    ('experiment_DynaQtable_Feb6_1543.py', 5, 14),  # n_bins=50
    ('experiment_DynaQtable_Feb2_0946.py', 5, 14),  # n_bins=100
    ('experiment_DynaQtable_Feb6_1544.py', 5, 14),  # n_bins=250
    # dmW, different num simulated experiences
    # ('experiment_DynaQNN_130_Feb10_2316.py', 0, 14),  # n_sim=0
    ('experiment_DynaQNN_130_Feb10_2317.py', 2, 14),  # n_sim=2
    ('experiment_DynaQNN_Feb5_1007.py', 5, 14),  # n_sim=5
    ('experiment_DynaQNN_Feb10_2300.py', 10, 14),  # n_sim=10
    ('experiment_DynaQNN_Feb10_2305.py', 12, 14),  # n_sim=12
    ('experiment_DynaQNN_Feb10_2302.py', 16, 14),  # n_sim=16
    ('experiment_DynaQNN_Feb10_2303.py', 20, 14),  # n_sim=20
]

exp_configs_legacy = [
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


#############################################################
# Configurations Above
#############################################################

def build_cmd(exp_file, num_sim, num_log):
    if '130' not in exp_file:
        log_file = '_'.join(['msg'] + exp_file.replace('.', '_').split('_')[1:4])
    else:
        log_file = '_'.join(['msg'] + exp_file.replace('.', '_').split('_')[1:5])
    type_agent = exp_file.split('_')[1]
    exp_list = ['python ./kdd-exps/' + exp_file + ' ' + str(i) for i in range(num_log)]
    if type_agent == 'DynaQNN':
        cmd_index = 'python ./' +'log_indexing_DynaQNN.py {log_file} {num_sim} {num_log} {num_proc}'.format(
            log_file=log_file, num_sim=num_sim, num_log=num_log, num_proc=n_process_idx
        )
    elif type_agent == 'DynaQtable':
        cmd_index = 'python ./' +'log_indexing_DynaQtable.py {log_file} {num_sim} {num_log} {num_proc}'.format(
            log_file=log_file, num_sim=num_sim, num_log=num_log, num_proc=n_process_idx
        )
    else:
        cmd_index = 'python ./' +'log_indexing_phiNN.py {log_file} {num_log} {num_proc}'.format(
            log_file=log_file, num_log=num_log, num_proc=n_process_idx
        )        
    return (exp_list, cmd_index, (log_file, num_log, num_log))

def build_legacy_cmd(prefix):
    log_file = prefix + '.log'
    exp_cmd = 'python ./kdd-exps/experiment_{}_legacy.py'.format(prefix)
    idx_cmd = 'python log_indexing_phiNN_legacy.py ./log/{}.log'.format(prefix)
    return (exp_cmd, idx_cmd)

def check_pid(pid):        
    """ Check For the existence of a unix pid. """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True

def run(cmd):
    p = subprocess.Popen(cmd, shell=True)
    p.wait()
    return

def load_dataframes(prefix, n_run, n=None):
    if n is None:
        n = n_run
    files = [prefix + "_{}.log".format(i) for i in range(n)]
    file_list = ['./log/index/' + prefix +'_x{}/'.format(n_run) +'index_'+file+'.csv' for file in files]
    df_list = [None]*n
    for i in range(n):
        t = time.time()
        df = pd.read_csv(file_list[i], delimiter=';', index_col=0)
        df.loc[:, 'start_ts'] = df['start_ts'].apply(lambda x: pd.to_datetime(x))
        df.set_index('start_ts', inplace=True)
        df['total_reward'] = df['tr_reward'] + df['op_cost']
        df_list[i] = df
        print "    Loaded",
        print files[i],
        print 'shape:',
        print df.shape,
        print "{:.2f} sec".format(time.time()-t)
    return df_list   

def get_step_reward(file_prefix, num_total, num_load):
    df_list = load_dataframes(file_prefix, num_total, num_load)
    # df_list = filter(lambda x: x.shape[0]==302400, df_list)
    # start = pd.to_datetime("2014-10-16 9:30:00")
    # end = pd.to_datetime("2014-10-21 9:30:00")
    delta = pd.Timedelta('2 seconds')

    step_reward = np.zeros(len(df_list))
    for i, df in enumerate(df_list):
        # df = df.loc[start:end]
        # print (i, df.shape[0])
        step = (df.index-df.index[0])/delta+1
        ts = df['total_reward'].cumsum()/step
        step_reward[i] = ts.iloc[-1]
    return step_reward

def log_step_reward(file, num_log, step_reward):
    with open('./log/index/{file}_x{num_log}/{file}.reward'.format(file=file, num_log=num_log), 'w') as reward_file:
        print>>reward_file, step_reward.tolist()

# Wait for previous run
while(True):
    if previous_pid is not None and check_pid(previous_pid):
        print datetime.now().strftime('[%Y-%m-%d %H:%M:%S]'),
        print "Proceses {} is running, retry in 600 seconds. (I'm {})".format(previous_pid, os.getpid())
        sleep(600)
    else:
        break    

runs = map(lambda x: build_cmd(*x), exp_configs)

# pool = Pool(n_process_exp)
# for exp_list, cmd_index, (log_file, num_log, num_log) in runs:
#     print datetime.now().strftime('[%Y-%m-%d %H:%M:%S]'),
#     print "Experiments start:"
#     pool.map(run, exp_list)
#     print datetime.now().strftime('[%Y-%m-%d %H:%M:%S]'),
#     print "Indexing log files:",
#     print cmd_index
#     run(cmd_index)
#     print datetime.now().strftime('[%Y-%m-%d %H:%M:%S]'),
#     print "Calculating rewards:"
#     step_reward = get_step_reward(log_file, num_log, num_log)
#     print "    {} sims".format(len(step_reward))
#     print "    mean {:.5f}, std {:.5f},".format(step_reward.mean(), step_reward.std())
#     print "    10% {:.5f}, 50% {:.5f}, 90% {:.5f},".format(*np.percentile(step_reward, [10, 50, 90]))
#     log_step_reward(log_file, num_log, step_reward)
# pool.close()

runs_legacy = map(lambda x: build_legacy_cmd(x[0]), exp_configs_legacy)
for (exp_cmd, idx_cmd) in runs_legacy:
    run(exp_cmd)
    run(idx_cmd)
    

