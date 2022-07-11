import collections
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import sys
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
sys.path.append(str(Path.cwd().joinpath('code')))
from pydub import AudioSegment
from pydub.playback import play
from scipy.signal import find_peaks


from src.utils.utils import GetSortedSpeciesCode

# -------------
TARGET_SPECIES = GetSortedSpeciesCode()
THRESHOLD = [0.26, 0.27, 0.41, 0.30, 0.45, 0.39, 0.43, 0.21, 0.36]
WIN_LEN = 1.0
OVERLAP = 0.75
HOP_LEN = WIN_LEN * (1 - OVERLAP)

# -------------
def showMelSpec(filePath, st:float, et:float):
  audioPath = Path.cwd().joinpath('data', 'raw', filePath)
  audio, sr = librosa.load(str(audioPath), sr=None, offset=st, duration=et-st)

  pcenTime = 0.06
  pcenGain = 0.8
  pcenBias = 10
  pcenPower = 0.25
  mel = librosa.feature.melspectrogram(
    y=audio, sr=sr, n_fft=1024, hop_length=512, 
    n_mels=128, fmin=1000, fmax=10000
  )
  image = librosa.pcen(
    mel * (2**31), sr=sr,
    time_constant=pcenTime, gain=pcenGain, bias=pcenBias, power=pcenPower
  )
  fig, ax = plt.subplots(figsize=(6, 6))
  librosa.display.specshow(
    image, y_axis='linear', x_axis='time',
    sr=sr, ax=ax, fmin=1000, fmax=10000
  )
  ax.set_title(audioPath.name + f'    {st} ~ {et}')
  plt.show(block=False)

def playAudio(filePath, st:float, et:float):
  audioPath = Path.cwd().joinpath('data', 'raw', filePath)
  audio = AudioSegment.from_file(str(audioPath))
  audio = audio[st*1000:(st+1)*1000]
  play(audio)

def findPeaks(df:pd.DataFrame):
  ## Get probability peak for each species
  allPeaks = collections.defaultdict(list)
  for i, sp in enumerate(TARGET_SPECIES):
    peaks, _ = find_peaks(df[sp], height=THRESHOLD[i])
    for p in peaks:
      allPeaks[p].append(sp)
  return allPeaks

def concatToLabel(file, st, et, labelList):
  labelDF = pd.read_csv(
    Path.cwd().joinpath('data', 'LABEL.csv'), header=0
  )
  dfDict = {l:1 for l in labelList if bool(l)}
  dfDict['file'] = file
  dfDict['start time'] = st
  dfDict['end time'] = et
  df = pd.DataFrame(dfDict, index=[0])
  
  labelDF = pd.concat([labelDF, df], ignore_index=True)
  labelDF.fillna(0, inplace=True)
  colnames = list(labelDF.columns)
  colnames = colnames[3:]
  labelDF = labelDF[(labelDF[colnames] != 0).any(axis=1)]
  labelDF.to_csv(Path.cwd().joinpath('data', 'LABEL.csv'), header=True, index=False)

def getTrueFalse(askStr, defaultVal):
  given = ''
  valid = False
  while not valid:
    given = input(askStr)
    if not given:
      given = defaultVal
    if given.lower() in ['y', 'yes']:
      return True
    elif given.lower() in ['n', 'no']:
      return False
    else:
      print('Invalid Input')

def strToLabel(speciesStr):
  res = [sp.upper() for sp in speciesStr.split(',')]
  return res

def ResultCorrectVisualsation():
  if len(THRESHOLD) != len(TARGET_SPECIES):
    raise ValueError('len of threshold must be same as the target species!')
  print(list(zip(TARGET_SPECIES, THRESHOLD)))

  ## Read single test file
  resProbDF = pd.read_csv(
    Path.cwd().joinpath('data', 'res_single.csv'), header=0,
    names=['file', 'st', 'et'] + TARGET_SPECIES
  )
  
  ## Get probability peak for each species
  allPeaks = findPeaks(resProbDF)

  for peakIndex, spList in allPeaks.items():
    predict = spList
    label = predict # Dafualt predicted label is true

    print(f'\033[1;31;43m {predict} \033[0;0m')    

    ### Play audio and show spectrogram repeatly
    showMelSpec(
        filePath = resProbDF.loc[peakIndex, 'file'], 
        st = resProbDF.loc[peakIndex, 'st'], 
        et = resProbDF.loc[peakIndex, 'et']
    )
    repeat = True
    while repeat:
      playAudio(
        filePath = resProbDF.loc[peakIndex, 'file'], 
        st = resProbDF.loc[peakIndex, 'st'], 
        et = resProbDF.loc[peakIndex, 'et']
      )
      repeat = getTrueFalse(f'\033[1;31;47m 重複播放片段？ (y)es / (n)o, 預設 no >> \033[0;0m', 'n')
    
    ### Correct the predicted label
    needCorrect = getTrueFalse(f'\033[1;31;47m 修改預測標籤？ (y)es / (n)o, 預設 no >> \033[0;0m', 'n')
    while needCorrect:
      label = input(f'\033[1;31;47m 物種？ (使用逗號分隔 "," | 若無物種請直接按Enter) \033[0;0m')
      label = strToLabel(label)
      print(f'\033[1;31;43m Label: {label} \033[0;0m')
      needCorrect = not getTrueFalse(f'\033[1;31;47m 修改完成？ (y)es / (n)o, 預設 yes >> \033[0;0m', 'y')

    ### Append label to dataset
    concatToLabel(
      file=resProbDF.loc[peakIndex, 'file'],
      st = resProbDF.loc[peakIndex, 'st'], 
      et = resProbDF.loc[peakIndex, 'et'],
      labelList=label
    )

# -------------
if __name__ == '__main__':
  ResultCorrectVisualsation()