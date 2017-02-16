import sys
sys.path.append('../../')  # add project home into search path
import pandas as pd
pd.set_option('mode.chained_assignment', None)
from sleep_control.integration import Emulation
from sleep_control.traffic_emulator import TrafficEmulator
from sleep_control.traffic_server import TrafficServer
from sleep_control.controller import QController
from rl.qnn_theano import QAgentNN
from rl.mixin import PhiMixin


class QAgentNNPhi(PhiMixin, QAgentNN):
    def __init__(self, **kwargs):
        super(QAgentNNPhi, self).__init__(**kwargs)


def set_up_data(f_name):
    print "Reading data...",
    session_df =pd.read_csv(
        filepath_or_buffer='../data/trace_dh3.dat',
        parse_dates=['startTime_datetime', 'endTime_datetime']
    )
    print "Complete!"
    return session_df


# Setting up data
session_df = set_up_data("../data/net_traffic_nonull_sample.dat")

# Setting up Emulation
print "Setting up Emulation environment..."
te = TrafficEmulator(session_df=session_df, time_step=pd.Timedelta(minutes=30), verbose=1)
ts = TrafficServer(verbose=2)
actions = [(True, None), (False, 'serve_all')]
phi_length = 3
range_state_slice = [(0, 10), (0, 10), (0, 10), (0, 1), (0, 1)]
agent = QAgentNNPhi(
        phi_length=phi_length,
        dim_state=(1, phi_length, 3+2),
        range_state=[[range_state_slice]*phi_length],
        actions=actions,
        learning_rate=0.01, reward_scaling=10, batch_size=100,
        freeze_period=50, memory_size=200, num_buffer=2,
        alpha=0.5, gamma=0.5, explore_strategy='epsilon', epsilon=0.02,
        verbose=2)
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

