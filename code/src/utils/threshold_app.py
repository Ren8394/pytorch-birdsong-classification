import collections
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pandas as pd
import seaborn as sn
import sys

from pathlib import Path
sys.path.append(str(Path.cwd().joinpath('code')))
from scipy.signal import find_peaks
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, fbeta_score
from tqdm import tqdm

from src.utils.utils import GetSortedSpeciesCode

# -------------
TARGET_SPECIES = GetSortedSpeciesCode()

# -------------
def findThreshold(pDF:pd.DataFrame, lDF:pd.DataFrame):
  thres = np.arange(0, 1, 0.05)
  maxThresDict = {sp:(0, 0, 0, 0, 0) for sp in TARGET_SPECIES}
  spThresDict = collections.defaultdict(list)
  for th in thres:
    thresDF = pd.DataFrame(columns=['file']+[s+'(P)' for s in TARGET_SPECIES]+TARGET_SPECIES)
    for i, f in tqdm(
      enumerate(pDF['file'].unique()), total=pDF['file'].unique().shape[0],
      desc=f'Threshold = {th:>.2f}',
      bar_format='{l_bar}{bar:32}{r_bar}{bar:-32b}'
    ):
      thresDF.loc[i, 'file'] = f
      thresDF.loc[i, TARGET_SPECIES] = lDF.loc[lDF['file'] == Path(f), TARGET_SPECIES].iloc[0, :].to_list()
      for sp in TARGET_SPECIES:
        peaks, _ = find_peaks(pDF.loc[pDF['file'] == f, f'{sp}(P)'], height=th)
        thresDF.loc[thresDF['file'] == f, f'{sp}(P)'] = len(peaks)
    thresDF.fillna(0, inplace=True)
    thresDF.loc[:, thresDF.columns != 'file'] = thresDF.loc[:, thresDF.columns != 'file'].astype(bool)
    for j, sp in enumerate(TARGET_SPECIES):
      accuracy = accuracy_score(
        y_true=thresDF.loc[:, sp].to_numpy(),
        y_pred=thresDF.loc[:, f'{sp}(P)'].to_numpy()
      ) # Accuracy
      f1 = fbeta_score(
        y_true=thresDF.loc[:, sp].to_numpy(),
        y_pred=thresDF.loc[:, f'{sp}(P)'].to_numpy(),
        beta=1,
        zero_division=0
      ) # F1
      precision = precision_score(
        y_true=thresDF.loc[:, sp].to_numpy(),
        y_pred=thresDF.loc[:, f'{sp}(P)'].to_numpy(),
        zero_division=0
      ) # Precision
      recall = recall_score(
        y_true=thresDF.loc[:, sp].to_numpy(),
        y_pred=thresDF.loc[:, f'{sp}(P)'].to_numpy(),
        zero_division=0
      ) # Recall
      spThresDict[sp].append([
        np.round(th, decimals=2), np.round(f1, decimals=4),
        np.round(precision, decimals=4), np.round(recall, decimals=4),
        np.round(accuracy, decimals=4)
      ])
      # if maxThresDict[sp][1] < f1:      # Use F1
      if maxThresDict[sp][4] < accuracy:  # Use Accuracy
        maxThresDict[sp] = (
          np.round(th, decimals=2), np.round(f1, decimals=4),
          np.round(precision, decimals=4), np.round(recall, decimals=4),
          np.round(accuracy, decimals=4)
        )
  return maxThresDict, spThresDict

def visualiseConfusionMatrix(pDF:pd.DataFrame, lDF:pd.DataFrame, maxDict:dict):
  thresDF = pd.DataFrame(columns=['file']+[s+'(P)' for s in TARGET_SPECIES]+TARGET_SPECIES)
  for i, f in tqdm(
    enumerate(pDF['file'].unique()), total=pDF['file'].unique().shape[0],
    bar_format='{l_bar}{bar:32}{r_bar}{bar:-32b}'
  ):
    thresDF.loc[i, 'file'] = f
    thresDF.loc[i, TARGET_SPECIES] = lDF.loc[lDF['file'] == Path(f), TARGET_SPECIES].iloc[0, :].to_list()
    for sp in TARGET_SPECIES:
      peaks, _ = find_peaks(pDF.loc[pDF['file'] == f, f'{sp}(P)'], height=maxDict[sp][0])
      thresDF.loc[thresDF['file'] == f, f'{sp}(P)'] = len(peaks)
  thresDF.fillna(0, inplace=True)

  ## Visualise - Confusion Matrix
  thresDF.loc[:, thresDF.columns != 'file'] = thresDF.loc[:, thresDF.columns != 'file'].astype(bool)
  _, axs = plt.subplots(3, 3, figsize=(12, 12), tight_layout=True)
  axs = axs.flatten()
  for j, sp in enumerate(TARGET_SPECIES):
    cm = confusion_matrix(
      y_true=thresDF[sp].to_numpy(),
      y_pred=thresDF[f'{sp}(P)'].to_numpy()
    )
    sn.heatmap(
      cm, annot=True, annot_kws={'size': 16}, fmt='d', cbar=False, ax=axs[j],
      cmap='coolwarm', linecolor='white', linewidths=1
    )
    axs[j].set_title(f'{sp} Threshold : {maxDict[sp][0]}')
    axs[j].set_xlabel('Pred')
    axs[j].set_ylabel('Label')
  plt.savefig(Path.cwd().joinpath('cm-acc.png'))

def countFileLabels(filePaths:Path):
  countDF = pd.DataFrame(columns=['file']+TARGET_SPECIES)
  for i, filePath in enumerate(filePaths):
    countDF.loc[i, 'file'] = Path('NrAudio', f'{filePath.stem}.wav')
    
    labelDF = pd.read_csv(filePath, sep='\t', names=['st', 'et', 'label'])
    labelDF['label'] = labelDF['label'].str.upper().replace(' ', '')
    labelDF = labelDF[labelDF['label'].str.contains('-S+', regex=True, na=False)]
    labelDF.reset_index(drop=True, inplace=True)
    labelDF['label'] = labelDF['label'].apply(lambda x: str(x).split('-')[0])
    labelDF = labelDF[labelDF['label'].apply(lambda x: x in TARGET_SPECIES)] # Select TARGET_SPECIES

    if labelDF.empty:
      continue
    
    vcDict = labelDF['label'].value_counts()
    for k, v in vcDict.items():
      countDF.loc[i, k] = v

  countDF.fillna(0, inplace=True)
  return countDF

def visualiseThreshold(tDict:dict):
  _, axs = plt.subplots(3, 3, figsize=(12, 12), tight_layout=True)
  axs = axs.flatten()
  for i, sp in enumerate(TARGET_SPECIES):
    scoreList = tDict[sp]
    thres = [s[0] for s in scoreList]
    f1 = [s[1] for s in scoreList]
    precision = [s[2] for s in scoreList]
    recall = [s[3] for s in scoreList]
    accuracy = [s[4] for s in scoreList]
    axs[i].set_ylim(0, 1)
    axs[i].set_xlim(0, 1)
    axs[i].plot(thres, f1, 'b-')
    axs[i].plot(thres, precision, 'r-')
    axs[i].plot(thres, recall, 'k-')
    axs[i].plot(thres, accuracy, 'y-')
    axs[i].set_title(sp)
  plt.savefig(Path.cwd().joinpath('thres.png'))

def ThresholdTest():
  sLabelPaths = sorted(Path.cwd().joinpath('data', 'raw', 'Label').glob('*.txt'))
  countDF = countFileLabels(sLabelPaths)

  ## Load Predict Probability CSV
  predictDF = pd.read_csv(Path.cwd().joinpath('data', 'TEST_APP.csv'), header=0)

  ## Find threshold for each species
  maxThresDict, spThresDict = findThreshold(predictDF, countDF)
  visualiseConfusionMatrix(predictDF, countDF, maxThresDict)
  visualiseThreshold(spThresDict)

# -------------
if __name__ == '__main__':
  ThresholdTest()
