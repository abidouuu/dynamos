import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from simu import config
from tqdm import tqdm
import random
from itertools import product
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
matplotlib.use("Agg")

workdir = os.path.dirname(os.path.abspath(__file__))
datadir=os.path.join(workdir, "data/skumanich")

for i in tqdm(range(4), desc="simus"):
    cfg=config(datadir=datadir, term='long', tfin=100000)
    cfg.nu=random.choice([0.1,1,10,100])
    data=cfg.run(save=False)
    cfg.write_config_file
    minimas=cfg.stat_analysis(data)
    cfg.plot_time(data=data, type='Bb', show=False)
    cfg.plot_time(data=data, type='kappa', show=False)
    cfg.plot_time(data=data, type='epsilon', show=False)