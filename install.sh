#!/bin/bash

# # this installs the right pip and dependencies for the fresh python
conda install ipython pip
conda install python=3.7

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

# dependencies for ax
pip install scipy botorch jinja2 pandas plotly

# install ax platform
pip3 install ax-platform

# for tensorboard
pip install tensorboard

pip install tqdm