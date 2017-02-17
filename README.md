# DeepNap
DeepNap is a deep reinforcement learning based sleeping control agent for base stations in mobile networks.
This repository is maintained review purpose of our paper: **DeepNap: Data-Driven Base Station Sleeping Operations through Deep Reinforcement Learning**

## Requirements and dependencies
All code are tested for `Ubuntu 14.04.4 LTS` with `CUDA 7.5.17`. The required Python packages are:
* Cython==0.25.2
* regex==2017.1.17
* jsonschema==2.5.1
* numpy==1.11.1
* scipy==0.17.0
* pandas==0.19.2
* scikit_learn==0.18.1
* hmmlearn==0.2.0
* Theano==0.8.2
* Lasagne==0.1
* matplotlib=1.5.1

To repeating experiments, >500G disk space is recommended to store all raw log files and indexed data. Also for speed condiderations, we recommend using multi-core CPU and GPU to fully leverage parallel execution.

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
Un-compress data
```shell
cd project_home/data/
tar xvf traces_kdd.tar.gz
```
Try running the mimi-experiment to test if installation is sucessful:
``shell
cd project_home/experiments/
python run_mini.py
``
You should start seeing logging outputs in shell.
