#!/bin/bash

conda install ipython pip
conda install python=3.7

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

pip install scipy botorch jinja2 pandas plotly optuna tensorboard tqdm



