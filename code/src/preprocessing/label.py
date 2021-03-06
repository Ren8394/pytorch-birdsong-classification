import numpy as np
import pandas as pd
import soundfile as sf
import sys

from pathlib import Path
sys.path.append(str(Path.cwd().joinpath('code')))
from tqdm import tqdm

from src.utils.utils import GetSortedSpeciesCode

# -------------
TARGET_SPECIES = GetSortedSpeciesCode() # 目標物種
WIN_LEN = 1.0                           # 滑移視窗長度
OVERLAP = 0.75                          # 滑移視窗重疊比例
HOP_LEN = WIN_LEN * (1 - OVERLAP)       # 依照滑移視窗與重疊比例計算每次滑移距離

# -------------
def audioTimeSegment(audioLen):
  """
    從 0 開始每一步滑移 {HOP_LEN} 直到 (訊號長度 - {WIN_LEN})
  """
  return list(np.around(np.arange(0, audioLen-WIN_LEN, HOP_LEN), decimals=6))

def segment(df:pd.DataFrame):
  """
    標籤切割演算法
    Reference: TBD
  """
  records = []  # start time as L and end time as R, and its index
  labels = []   # label
  for i, x in df.iterrows():
    records.append([x['start time'], 'L', i]) 
    records.append([x['end time'], 'R', i])
    labels.append(x['label'])
  ## Sort by time
  records = sorted(records)

  overlap = []
  res = []
  for j, record in enumerate(records):
    if record[1] == 'L': 
      ## Overlap means get L but previous L' aren't finished by its R'
      if overlap:
        labelList = [label for k, label in enumerate(labels) if k in overlap]
        labelList = set(labelList)  # Use set to remove identical label
        res.append([records[j-1][0], record[0], labelList])
        overlap.append(record[2])
      else:
        overlap.append(record[2])
    else:
      labelList = [label for k, label in enumerate(labels) if k in overlap]
      labelList = set(labelList)
      res.append([records[j-1][0], record[0], labelList])
      overlap.remove(record[2])
  return res

# ------------- Segment
def labelType(filePaths:Path):
  """ Classifier
    1. 篩選 Song 標籤 (*-S, *-S1, ....)
    2. 依照標籤進行切割 (segment -> func)
    3. 切割後進行滑移視窗
    4. 保留 duration 大於0.5秒部分
  """
  segDF = pd.DataFrame(columns=['file', 'start time', 'end time', 'label'])
  for filePath in tqdm(filePaths, bar_format='{l_bar}{bar:32}{r_bar}{bar:-32b}'):
    ## Preprocess
    labelDF = pd.read_csv(filePath, sep='\t', names=['start time', 'end time', 'label'])
    labelDF['label'] = labelDF['label'].str.upper().replace(' ', '')
    labelDF = labelDF[labelDF['label'].str.contains('-S+', regex=True, na=False)]
    labelDF.reset_index(drop=True, inplace=True)
    labelDF['label'] = labelDF['label'].apply(lambda x: str(x).split('-')[0])
    if labelDF.empty:
      continue

    ## Interval segmentment
    tempDF = pd.DataFrame(segment(labelDF), columns=['start time', 'end time', 'label'])
    source = sf.SoundFile(Path.cwd().joinpath('data', 'raw', 'NrAudio', f'{filePath.stem}.wav'))
    tempDF['end time'] = tempDF['end time'].apply(lambda x: min(x, source.frames / source.samplerate))
    tempDF['file'] = Path('NrAudio', f'{filePath.stem}.wav')
    tempDF = tempDF[['file', 'start time', 'end time', 'label']]

    ## Sliding
    smallDF = tempDF[tempDF['end time'] - tempDF['start time'] <= WIN_LEN]    # Duration less and equal than {WIN_LEN}
    largeDF = tempDF[tempDF['end time'] - tempDF['start time'] > WIN_LEN]     # Duration greater than {WIN_LEN}
    for _, x in largeDF.iterrows():
      st, et = x['start time'], x['end time']-WIN_LEN
      slideList = []
      ### Slide signal duration longer than {WIN_LEN} and concat it into the {smallDF}
      while st <= et:
        slideList.append([x['file'], np.around(st, decimals=6), np.around(st+WIN_LEN, decimals=6), x['label']])
        st += HOP_LEN

      smallDF = pd.concat(
        [smallDF, pd.DataFrame(slideList, columns=['file', 'start time', 'end time', 'label'])],
        ignore_index=True
      )
    smallDF = smallDF[smallDF['end time'] - smallDF['start time'] > 0.5] # Only duration > 0.5 are accept
    smallDF.sort_values(by=['file', 'start time', 'end time'], inplace=True)
    smallDF['label'] = smallDF['label'].apply(lambda x: sorted(x))

    ## Concatenate
    segDF = pd.concat([segDF, smallDF], ignore_index=True)
  return segDF

# ------------- Window
# Not Use
def windowType(filePaths:Path):
  winDF = pd.DataFrame(columns=['file', 'start time', 'end time'] + TARGET_SPECIES)
  for filePath in tqdm(filePaths, bar_format='{l_bar}{bar:32}{r_bar}{bar:-32b}'):
    ## Construct temporary DataFrames
    tempDF = pd.DataFrame(columns=['file', 'start time', 'end time'] + TARGET_SPECIES)
    source = sf.SoundFile(Path.cwd().joinpath('data', 'raw', 'NrAudio', f'{filePath.stem}.wav'))
    tempDF['start time'] = audioTimeSegment(source.frames / source.samplerate)
    tempDF['end time'] = tempDF['start time'] + 1
    tempDF['file'] = Path('NrAudio', f'{filePath.stem}.wav')
    
    ## Fill in the label
    labelDF = pd.read_csv(filePath, sep='\t', names=['st', 'et', 'label'])
    labelDF['label'] = labelDF['label'].str.upper().replace(' ', '')
    labelDF = labelDF[labelDF['label'].str.contains('-S+', regex=True, na=False)]
    labelDF.reset_index(drop=True, inplace=True)
    labelDF['label'] = labelDF['label'].apply(lambda x: str(x).split('-')[0])
    labelDF = labelDF[labelDF['label'].apply(lambda x: x in TARGET_SPECIES)] # Select TARGET_SPECIES
    if labelDF.empty:
      continue

    for _, x in labelDF.iterrows():
      s = np.ceil((x['st'] - WIN_LEN) // HOP_LEN) 
      e = np.floor((x['et'] - WIN_LEN) // HOP_LEN)
      ### Avoid s greater than e because of ceil and floor operator
      ### s in the start index, e is the end index
      if s > e:
        s, e = e, s
      ### Fill in
      tempDF.loc[s:e, x['label']] = int(1)

    ## Concat with winDF
    winDF = pd.concat(
      [winDF, tempDF], ignore_index=True
    )
  winDF.fillna(0, inplace=True)
  return winDF

# ------------- AutoEncoder
def aeType(filePaths:Path):
  """ Autoencoder
    讀取音檔長度, 並依照 {WIN_LEN, OVERLAP, HOP_LEN}切割 (audioTimeSegment -> func)
  """
  ae = []
  for filePath in tqdm(filePaths, bar_format='{l_bar}{bar:32}{r_bar}{bar:-32b}'):
    tempDF = pd.DataFrame(columns=['file', 'start time', 'end time'])
    source = sf.SoundFile(Path.cwd().joinpath('data', 'raw', 'NrAudio', f'{filePath.stem}.wav'))
    tempSts = audioTimeSegment(source.frames / source.samplerate)
    for tempSt in tempSts:
      ae.append([Path('NrAudio', f'{filePath.stem}.wav'), tempSt, tempSt+WIN_LEN])
      
  aeDF = pd.DataFrame(ae, columns=['file', 'start time', 'end time'])
  return aeDF

# -------------
def ConcatLabel():
  """
    {sLabelPaths} 為私人錄音檔標籤路徑
    {oLabelPaths} 為OpenSource音檔標籤路徑
    {audioFilePaths} 所有已降噪音檔路徑
    1. 將標籤依照 Classifier 需求創建資料集 (label_seg.csv)
    2. 將標籤依照 Autoencoder 需求創建資料集 (label_ae.csv)
  """
  sLabelPaths = sorted(Path.cwd().joinpath('data', 'raw', 'Label').glob('*.txt'))
  oLabelPaths = sorted(Path.cwd().joinpath('data', 'raw', 'Opensource').glob('*.txt'))
  audioFilePaths = sorted(Path.cwd().joinpath('data', 'raw', 'NrAudio').glob('*.wav'))
  segDF = labelType(sLabelPaths + oLabelPaths)
  aeDF = aeType(audioFilePaths)
  segDF.to_csv(Path.cwd().joinpath('data', 'label_seg.csv'), header=True, index=False)
  aeDF.to_csv(Path.cwd().joinpath('data', 'label_ae.csv'), header=True, index=False)

# -------------
if __name__ == '__main__':
  ConcatLabel()