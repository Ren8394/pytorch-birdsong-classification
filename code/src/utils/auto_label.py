import collections
import librosa
import numpy as np
import noisereduce as nr
import pandas as pd
import soundfile as sf
import sys
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
sys.path.append(str(Path.cwd().joinpath('code')))
from scipy.signal import find_peaks
from torch.utils.data import DataLoader
from tkinter import *
from tkinter.filedialog import askopenfilenames, askopenfilename
from tqdm import tqdm

from src.network.dataset import BirdsongDataset
from src.network.network import AutoEncoderClassifer
from src.utils.utils import GetSortedSpeciesCode

# -------------
TARGET_SPECIES = GetSortedSpeciesCode()                               # 取得目標物種list
THRESHOLD = [0.26, 0.27, 0.41, 0.30, 0.45, 0.39, 0.43, 0.21, 0.36]    # 各物種機率peak threshold

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

def noiseReduce(filePaths:Path):
  outputFilePaths = []
  for filePath in filePaths:
    filePath = Path(filePath)
    audio, sr = librosa.load(str(filePath), sr=None)
    audio = audio.T
    audio = librosa.util.normalize(audio)
    audio = nr.reduce_noise(y=audio, sr=sr, n_jobs=-1)
    outputFilePath = Path.cwd().joinpath('data', 'raw', 'NrAudio', f'{filePath.stem}.wav')
    outputFilePath.parent.mkdir(parents=True, exist_ok=True)
    sf.write(outputFilePath, audio, sr, 'PCM_24')
    outputFilePaths.append(outputFilePath)
  return outputFilePaths

def audioTimeSegment(audioLen):
  return list(np.around(np.arange(0, audioLen-WIN_LEN, HOP_LEN), decimals=6))

def genTestFile(filePath:Path):
  source = sf.SoundFile(filePath)
  tempDF = pd.DataFrame(audioTimeSegment(source.frames / source.samplerate), columns=['start time'])
  tempDF['end time'] = tempDF['start time'] + 1
  tempDF['file'] = Path('NrAudio', filePath.name)
  tempDF['code'] = [[0] * len(TARGET_SPECIES)] * tempDF.shape[0]
  tempDF = tempDF[['file', 'start time', 'end time', 'code']]
  tempDF.to_csv(Path.cwd().joinpath('data', 'tempTest.csv'), header=True, index=False)
  return Path.cwd().joinpath('data', 'tempTest.csv')

def genTestDataloader(filePath:Path):
  testingDataloader = DataLoader(
    BirdsongDataset(filePath, needAugment=False, needLabel=False), 
    batch_size=8, shuffle=False, num_workers=6, pin_memory=True
  )
  return testingDataloader

def findProbPeak(df:pd.DataFrame):
  allPeaks = collections.defaultdict(list)
  for i, sp in enumerate(TARGET_SPECIES):
    peaks, _ = find_peaks(df[sp], height=THRESHOLD[i])
    for p in peaks:
      allPeaks[p].append(sp)
  return allPeaks

def concatToLabel(file, peakDict):
  labelDF = pd.read_csv(Path.cwd().joinpath('data', 'label_auto.csv'), header=0)
  dfList = []
  for peakIndex, spList in peakDict.items():
    dfDict = {sp:1 for sp in spList if bool(sp)}
    dfDict['file'] = file
    dfDict['start time'] = peakIndex * HOP_LEN
    dfDict['end time'] = peakIndex * HOP_LEN + WIN_LEN
    dfList.append(dfDict)
  df = pd.DataFrame.from_records(dfList)
  df.sort_values(by=['file', 'start time', 'end time'], inplace=True)
  labelDF = pd.concat([labelDF, df], ignore_index=True)
  labelDF.fillna(0, inplace=True)
  colnames = list(labelDF.columns)
  colnames = colnames[3:]
  labelDF = labelDF[(labelDF[colnames] != 0).any(axis=1)]
  labelDF.drop_duplicates(subset=['file', 'start time', 'end time'], inplace=True)
  labelDF.to_csv(Path.cwd().joinpath('data', 'label_auto.csv'), header=True, index=False)

def AutoLabel():
  """
    1. 讀取自動測試音檔與模型weight
    2. 將選取音檔降噪
    3. 自動測試抓取各音檔機率peak並產生自動標籤
  """
  if len(THRESHOLD) != len(TARGET_SPECIES):
    raise ValueError('len of threshold must be same as the target species!')
  print(list(zip(TARGET_SPECIES, THRESHOLD)))
  
  ## Setting
  root = Tk()
  root.withdraw()
  audioPaths = askopenfilenames(
    title='Select Audio Files: ',
    initialdir=Path.cwd().joinpath('data'),
    filetypes=(('Audio files', '*.mp3, *.wav'))
  )
  weightPath = askopenfilename(
    title='Select Model Weight File', 
    initialdir=Path.cwd().joinpath('model'),
    filetypes=(('Weight files', '*.pth'),)
  )
  nrAudioPaths = noiseReduce(audioPaths)          # 降噪
  root.destroy()

  ## model
  model = AutoEncoderClassifer(numberOfClass=len(TARGET_SPECIES)).to(DEVICE)
  model.load_state_dict(
    torch.load(weightPath, map_location=torch.device(DEVICE))
  )

  ## Test and Label
  for nrPath in tqdm(nrAudioPaths):
    testFilePath = genTestFile(nrPath)                  # 產生測試檔案
    testDataloader = genTestDataloader(testFilePath)  
    predicts = test(model, testDataloader)    
    Path.unlink(testFilePath)                           # 刪除測試檔案
    
    predicts = np.array(np.reshape(predicts, (-1, len(TARGET_SPECIES))))
    predDF = pd.DataFrame(predicts, columns=TARGET_SPECIES)
    peakDict = findProbPeak(predDF)
    concatToLabel(Path('NrAudio', nrPath.name), peakDict)

# -------------
if __name__ == '__main__':
  AutoLabel()