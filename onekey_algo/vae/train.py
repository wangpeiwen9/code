# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2022/7/12
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2022 All Rights Reserved.
import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from onekey_core.models.vaes import *
from onekey_algo.vae.experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from onekey_algo.vae.dataset import VAEDataset
from pytorch_lightning.plugins import DDPPlugin


FlatVAE(f_len, 10, kernel_size=5, stride=4)