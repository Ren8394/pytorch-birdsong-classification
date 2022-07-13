import ast
import numpy as np
import pandas as pd
import torch

from pathlib import Path

# -------------
"""
從 SPECIES.csv 取出目標物種
"""
def GetSortedSpeciesCode():
  df = pd.read_csv(Path.cwd().joinpath('data', 'SPECIES.csv'), header=0)
  res = df.loc[df['Target'], 'Code'].tolist()  
  return res

""" (未使用)
從 STATION.csv 取出測站
"""
def GetStationList():
  df = pd.read_csv(Path.cwd().joinpath('data', 'STATION.csv'), header=0)
  res = df['station'].tolist()
  return res

"""
依照 {source} 將 {target} 進行 One-hot encoding
e.g. 
  source = ['A', 'B', 'C']
  target = 'B'
  >> onthot = [0, 1, 0]
"""
def OneHotEncoding(targets, source):
  onehot = [0] * len(source)
  for t in targets:
    onehot[source.index(t)] = int(1)
  return onehot

""" Imbalance Weight 計算
1. {filePath} 目標資料集路徑
2. {weightType} 不同計算方式選擇
  ins: 倒數
  isns: 倒數開根號
  xgboost: xgboost
  ens: Efficient Number of Samples
  none: 不設定
  ----
  > Class-Balanced Loss Based on Effective Number of Samples (2019)
  ----
"""
def CalculateImbalanceWeight(filePath, weightType='ens'):
  df = pd.read_csv(filePath, header=0)
  labels = []
  for _, x in df.iterrows():
    labels.append(ast.literal_eval(x['code']))
  df = pd.DataFrame.from_records(labels)
  classCounts = df.sum().to_list()
  ## Calculate different type
  if weightType == 'ins':
    weights = [ 1 / (c + 1e-5) for c in classCounts]
  elif weightType == 'isns':
    weights = [ 1 / np.sqrt(c + 1e-5) for c in classCounts]
  elif weightType == 'xgboost':
    weights = [ (sum(classCounts) - c) / (c + 1e-5) for c in classCounts]
  elif weightType == 'ens':
    beta = (len(df) - 1) / len(df)
    weights = [ (1 - beta) / (1 - np.power(beta, c)) for c in classCounts]
    sumOfWeights = sum(weights)
    weights = [ (w / sumOfWeights) * len(classCounts) for w in weights]
  elif weightType == 'none':
    return None
  else:
    raise ValueError(f'{weightType} type is not supported.\nThe supported types are ins, isns, xgboost, ens, and none.\n')
  return torch.as_tensor(weights).float()

# -------------
if __name__ == '__main__':
  print(
    CalculateImbalanceWeight(Path.cwd().joinpath('data', 'aec_train.csv'), 'ens')
  )
  print(GetSortedSpeciesCode())
  print(GetStationList())
