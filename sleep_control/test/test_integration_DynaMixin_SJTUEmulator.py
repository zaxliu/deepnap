import sys
sys.path.append('../../')  # add project home into search path
import time

import pandas as pd

from sleep_control.integration import Emulation
from sleep_control.traffic_emulator import TrafficEmulator
from sleep_control.traffic_server import TrafficServer
from sleep_control.controller import QController
from sleep_control.env_models import SJTUModel

from rl.qtable import QAgent
from rl.qnn_theano import QAgentNN
from rl.mixin import DynaMixin

pd.set_option('mode.chained_assignment', None)


# Define helper wrapper classes
class DynaQAgent(DynaMixin, QAgent):
    def __init__(self, **kwargs):
        super(DynaQAgent, self).__init__(**kwargs)

class DynaQAgentNN(DynaMixin, QAgentNN):
    def __init__(self, **kwargs):
        super(DynaQAgentNN, self).__init__(**kwargs)


# load from processed data
session_df =pd.read_csv(
    filepath_or_buffer='../../data/trace_dh3.dat',
    parse_dates=['startTime_datetime', 'endTime_datetime']
)

# Setting up Emulation
print "Setting up Emulation environment..."

# Parameters
# emulator
time_step = pd.Timedelta(seconds=2)  # emulator time step
# agent
actions = [(True, None), (False, 'serve_all')]
# env Model
model_type, traffic_window_size = 'IPP', 50
stride, n_iter, adjust_offset = 2, 3, 1e-22
eval_period, eval_len = 4, 100
n_belief_bins, max_queue_len = 5, 20
Rs, Rw, Rf, Co, Cs = 1.0, -1.0, -10.0, -5.0, -0.5
num_sim = 10

# Build entities
rewarding = {'serve': Rs, 'wait': Rw, 'fail': Rf}
te = TrafficEmulator(session_df=session_df, time_step=time_step,
                     rewarding=rewarding,
                     verbose=1)

ts = TrafficServer(verbose=2, cost=(Co, Cs))

traffic_params = (model_type, traffic_window_size,
                  stride, n_iter, adjust_offset,
                  eval_period, eval_len,
                  n_belief_bins)
queue_params = (max_queue_len,)
reward_params = (Rs, Rw, Rf, Co, Cs, None)
env_model = SJTUModel(traffic_params, queue_params, reward_params, verbose=1)

agent = DynaQAgent(env_model=env_model, num_sim=num_sim,
                   actions=actions, alpha=0.5, gamma=0.5,
                   explore_strategy='epsilon', epsilon=0.1,
                   verbose=2
        )
#agent = QAgentNN(dim_state=(1, 1, 3), range_state=((((0, 10), (0, 10), (0, 10),),),),
#                 learning_rate=0.01, reward_scaling=10, batch_size=100, freeze_period=50, memory_size=200, num_buffer=2,
#                 actions=actions, alpha=0.5, gamma=0.5, explore_strategy='epsilon', epsilon=0.1,
#                 verbose=2
#                 )
c = QController(agent=agent)
emu = Emulation(te=te, ts=ts, c=c)

print "Emulation starting"
print
# run...
while emu.te.epoch is not None:
    # log time
    print "Epoch {}, ".format(emu.epoch),
    left = emu.te.head_datetime + emu.te.epoch*emu.te.time_step
    right = left + emu.te.time_step
    print "{} - {}".format(left.strftime("%Y-%m-%d %H:%M:%S"), right.strftime("%Y-%m-%d %H:%M:%S"))
    emu.step()
    print

