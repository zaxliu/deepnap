import os
import time
from time import sleep
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Pool
import subprocess

#############################################################
# Configuration Area
#############################################################
previous_pid = None  # Unix PID of previous run scripts
n_process_exp = 1
n_process_idx = 1
exp_configs = [
    # (file, num_sim, num_log)
    ('experiment_DynaQNN_Feb15_2200_mini.py', 1, 2)
]

#############################################################
# 
#############################################################
def build_cmd(exp_file, num_sim, num_log):
    log_file = '_'.join(['msg'] + exp_file.split('_')[1:4])
    type_agent = exp_file.split('_')[1]
    exp_list = ['python ./' + exp_file + ' ' + str(i) for i in range(num_log)]
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
        print df.shape,
        print files[i],
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
        df = df.loc[start:end]
        print (i, df.shape[0])
        step = (df.index-df.index[0])/delta+1
        ts = df['total_reward'].cumsum()/step
        step_reward[i] = ts.iloc[-1]
    return step_reward

def log_step_reward(file, num_log, step_reward):
    with open('./log/index/''{file}_x{num_log}/{file}.reward'.format(file=file, num_log=num_log), 'w') as reward_file:
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

pool = Pool(n_process_exp)
for exp_list, cmd_index, (log_file, num_log, num_log) in runs:
    print datetime.now().strftime('[%Y-%m-%d %H:%M:%S]'),
    print "Experiments start:"
    pool.map(run, exp_list)
    print datetime.now().strftime('[%Y-%m-%d %H:%M:%S]'),
    print "Indexing log files:",
    print cmd_index
    run(cmd_index)
    print datetime.now().strftime('[%Y-%m-%d %H:%M:%S]'),
    print datetime.now().strftime('[%Y-%m-%d %H:%M:%S]'),
    print "Calculation rewards:"
    step_reward = get_step_reward(log_file, num_log, num_log)
    print step_reward
    log_step_reward(log_file, num_log, step_reward)
pool.close()

