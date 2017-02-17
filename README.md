# DeepNap
@Author Jingchu Liu, @Date Feb 17 2017

DeepNap is a deep reinforcement learning based sleeping control agent for base stations in mobile networks. This repository is maintained for review purposes of our paper: **DeepNap: Data-Driven Base Station Sleeping Operations through Deep Reinforcement Learning**

## Requirements and Dependencies
All code are tested for `Ubuntu 14.04.4 LTS` with `CUDA 7.5.17`. The required Python packages are listed in ``requirements.txt``. ```ipython Notebook``` is also required for visualization.

## Setup
Clone the code and data from Github:
```shell
git clone https://github.com/zaxliu/deepnap.git
```
Installing Python dependencies (recommend using virtual environment):
```shell
cd project_home/
pip install -r requirements
```
Download and unzip data (need Internet connectivity to download data ~300MB)
```shell
cd project_home/
python setup.py
```
Try running the mimi-experiment to test if installation is sucessful:
```shell
cd project_home/experiments/
python run_mini.py
```
You should start seeing logging outputs in shell.


## Reproducing Experimental Results
We provide code and data to reproduce the experimental results (figures and tables) in our paper.

Note you **_don't have to re-run_** all the experiments to see results. The setup routine will automatically download necessary datasets for visualization. You can find and reproduce all figures and tables used in our paper with this [ipython notebook](https://github.com/zaxliu/deepnap/blob/master/experiments/paper_figure_tables.ipynb).

If you **_do want to repeat all experiments_**, we also provide a [single script](https://github.com/zaxliu/deepnap/blob/master/experiments/run_full.py) to re-run all the experiments that we tested. Note as this process will produce a large amount of log and intermediate data files, we recommend **>500GB** disk space researved for testing. Disk requirements can be relaxed to **20GB** if you can manually delete all `.log` and `index_*.log.csv` files after you see the corresponding ``.reward`` file.

Also for speed considerations, we recommend using multi-core CPU, GPU, and >8G memory to fully leverage parallel execution. And be warned experiments are time-consuming - each experiment may take hours to finish. So be patient and have a drink, maybe.

