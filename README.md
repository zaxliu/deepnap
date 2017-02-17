# DeepNap
DeepNap is a deep reinforcement learning based sleeping control agent for base stations in mobile networks.

This repository is maintained review purpose of our paper: **DeepNap: Data-Driven Base Station Sleeping Operations through Deep Reinforcement Learning**

## Requirements and Dependencies
All code are tested for `Ubuntu 14.04.4 LTS` with `CUDA 7.5.17`. The required Python packages are listed in ``requirements.txt``. ```iPython Notebook``` is also required for visualization.

## Setup
Clone the code and data from Github:
```shell
git clone https://github.com/zaxliu/deepnap.git
```
Installing Python dependencies:
```shell
cd project_home/
pip install -r requirements
```
Download and unzip data:
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

Note you **_don't have to re-run_** all the experiments to see results. We provide proccessed experimental results (```.reward``` files). They can be conveniently loaded by the Ipython Notebooks for visualization purpose. See [Visualizting Results](#visualizing-results) section.

If you **_do want to repeat all experiments_**, **>500GB** disk space is recommended to store all raw log files and indexed data, and **20GB** is recommended if you can manually delete all `.log` and `index_*.log.csv` files after you see result outputs.

Also for speed considerations, we recommend using multi-core CPU, GPU, and >8G memory to fully leverage parallel execution. And be warned experiments are time-consuming - each experiment may take hours to finish. So be patient and have a drink, maybe.

### Visualizing Results
### Fully Reproducing Experiments

## Implementation Details

