import librosa
import noisereduce as nr
import numpy as np
import pandas as pd
import soundfile as sf
import sys
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
sys.path.append(str(Path.cwd().joinpath('code')))
from tqdm import tqdm
from torch.utils.data import DataLoader
from tkinter import *
from tkinter.filedialog import askopenfilename

from src.network.network import AutoEncoderClassifer
from src.network.dataset import BirdsongDataset
from src.utils.utils import GetSortedSpeciesCode

# -------------
TARGET_SPECIES = GetSortedSpeciesCode()
WIN_LEN = 1.0
OVERLAP = 0.75
HOP_LEN = WIN_LEN * (1 - OVERLAP)

if torch.cuda.is_available():
  DEVICE = torch.device('cuda:0')
  torch.backends.cudnn.benchmark = True
else:
  DEVICE = torch.device('cpu')

# -------------
def test(model, dataloader):
  model.eval()

  predicts = []
  with torch.no_grad():
    for _, (inputs, _) in tqdm(
      enumerate(dataloader), total=len(dataloader), bar_format='{l_bar}{bar:32}{r_bar}{bar:-32b}'
    ):
      inputs = inputs.to(DEVICE)
      outputs = nn.functional.sigmoid(model(inputs))
      predicts.extend(outputs.cpu().numpy())
  return predicts

def audioTimeSegment(audioLen):
  return list(np.around(np.arange(0, audioLen-WIN_LEN, HOP_LEN), decimals=6))

def noiseReduce(filePath):
  audio, sr = librosa.load(str(filePath), sr=None)
  audio = audio.T
  audio = librosa.util.normalize(audio)
  audio = nr.reduce_noise(y=audio, sr=sr, n_jobs=-1)
  outputFilePath = Path.cwd().joinpath('data', 'raw', 'NrAudio', f'{filePath.stem}.wav')
  outputFilePath.parent.mkdir(parents=True, exist_ok=True)
  sf.write(outputFilePath, audio, sr, 'PCM_24')

  return outputFilePath

def ExcuteSingleTestingProcess():
  ## Setting
  """
    1. 選擇 model weight
    2. 選擇測試音檔
    3. 音檔降噪存回 NrAudio
  """
  root = Tk()
  root.withdraw()
  weightPath = askopenfilename(title='Choose The File Of Model Weight', initialdir=Path.cwd().joinpath('model'))
  audioPath = askopenfilename(title='Select Testing Audio', initialdir=Path.cwd().joinpath('data'))
  nrPath = noiseReduce(Path(audioPath))
  root.destroy()

  ## model
  model = AutoEncoderClassifer(numberOfClass=len(TARGET_SPECIES)).to(DEVICE)
  model.load_state_dict(
    torch.load(weightPath, map_location=torch.device(DEVICE))
  )
  
  ## Temp File
  source = sf.SoundFile(nrPath)
  tempDF = pd.DataFrame(
    audioTimeSegment(source.frames / source.samplerate), columns=['start time']
  )
  tempDF['end time'] = tempDF['start time'] + 1
  tempDF['file'] = Path('NrAudio', nrPath.name)
  tempDF['code'] = [[0] * len(TARGET_SPECIES)] * tempDF.shape[0]
  tempDF = tempDF[['file', 'start time', 'end time', 'code']]
  tempDF.to_csv(Path.cwd().joinpath('data', 'tempTest.csv'), header=True, index=False)

  ## Test
  testingDataloader = DataLoader(
    BirdsongDataset(
      Path.cwd().joinpath('data', 'tempTest.csv'), 
      needAugment=False, 
      needLabel=False
    ), batch_size=8, shuffle=False, num_workers=6, pin_memory=True
  )
  predicts = test(model, testingDataloader)
  predicts = np.array(np.reshape(predicts, (-1, len(TARGET_SPECIES))))
  Path.unlink(Path.cwd().joinpath('data', 'tempTest.csv'))

  ## Result
  resDF = pd.concat(
    [
      tempDF[['file', 'start time', 'end time']], 
      pd.DataFrame(predicts, columns=[f'{sp}(P)' for sp in TARGET_SPECIES])
    ], axis=1, ignore_index=False
  )

  ## Save Result
  resDF.to_csv(Path.cwd().joinpath('data', 'res_single.csv'), header=True, index=False)

# -------------
if __name__ == '__main__':
  ExcuteSingleTestingProcess()
