import configparser
import numpy as np
import pandas as pd

from pathlib import Path
from tkinter import *
from tkinter.filedialog import askopenfilenames

from src.dataset import BirdsongDataset
from src.network import AutoEncoderClassifer
from src.utils import GetSortedSpeciesCode

# -------------
config = configparser.ConfigParser()
config.read(str(Path.cwd().parent.joinpath('setting', 'config.ini')))

WIN_LEN = config['Window'].getint('Length')
HOP_LEN = WIN_LEN * (1 - config['Window'].getfloat('Overlap'))
TARGET_SPECIES = GetSortedSpeciesCode(Path.cwd().parent.joinpath('setting', 'SPECIES.csv'))

# -------------
if __name__ == '__main__':
  pass