import ast
import numpy as np
import pandas as pd
import sys

from pathlib import Path
sys.path.append(str(Path.cwd().joinpath('code')))

from src.utils.utils import GetSortedSpeciesCode, OneHotEncoding

# -------------
TARGET_SPECIES = GetSortedSpeciesCode()
OPENSOURCE = ['XC', 'ML']

# -------------
def filterTarget(l):
  res = []
  for x in l:
    if x in TARGET_SPECIES:
      res.append(x)
  return res

def dataSplitAEC():
  ## Load Window Label (file, start time, end time, TARGET_SPECIES) >> only station
  winDF = pd.read_csv(Path.cwd().joinpath('data', 'LABEL.csv'), header=0)
  if not winDF.empty:
    winDF = winDF.loc[winDF[TARGET_SPECIES].any(axis=1), :]                   # Filter none label
    winDF['code'] = winDF[TARGET_SPECIES].apply(lambda x: x.tolist(), axis=1) # Concat to one hot encoding label
    winDF = winDF[['file', 'start time', 'end time', 'code']]

  ## Load Segment Label (file, start time, end time, label) >> station and opensource
  segDF = pd.read_csv(Path.cwd().joinpath('data', 'LABEL_SEG.csv'), header=0) 
  segDF['code'] = segDF['label'].apply(lambda x: filterTarget(ast.literal_eval(x)))                           # Filter TARGET_SPECIES
  segDF['code'] = segDF['code'].apply(lambda x: pd.NA if len(x) == 0 else OneHotEncoding(x, TARGET_SPECIES))  # Onehot encoding
  segDF = segDF[['file', 'start time', 'end time', 'code']]
  segDF.dropna(inplace=True)

  ## Combine to one dataset
  if not winDF.empty:
    df = pd.concat([winDF, segDF], ignore_index=True)
  else:
    df = segDF.copy()

  ## Split <train, validate, test> = <0.8 : 0.1 : 0.1>
  trainDF, validateDF, testDF = np.split(
    df.sample(frac=1), [int(0.8 * len(df)), int(0.9 * len(df))]
  )
  trainDF.to_csv(Path.cwd().joinpath('data', 'aec_train.csv'), header=True, index=False)
  validateDF.to_csv(Path.cwd().joinpath('data', 'aec_validate.csv'), header=True, index=False)
  testDF.to_csv(Path.cwd().joinpath('data', 'aec_test.csv'), header=True, index=False)

def dataSplitAE():
  ## Load Window Label (file, start time, end time, TARGET_SPECIES) >> only station
  aeDF = pd.read_csv(Path.cwd().joinpath('data', 'LABEL_AE.csv'), header=0)
  trainDF, validateDF = np.split(aeDF.sample(frac=1), [int(0.9 * len(aeDF))])
  trainDF.to_csv(Path.cwd().joinpath('data', 'ae_train.csv'), header=True, index=False)
  validateDF.to_csv(Path.cwd().joinpath('data', 'ae_validate.csv'), header=True, index=False)

def testAppData():
  aeDF = pd.read_csv(Path.cwd().joinpath('data', 'LABEL_AE.csv'), header=0)
  segDF = pd.read_csv(Path.cwd().joinpath('data', 'LABEL_SEG.csv'), header=0)
  fnames = ['Audio/'+p.split('/')[1] for p in segDF['file'].unique()]
  appDF = aeDF.loc[aeDF['file'].isin(fnames), :].copy()
  appDF['file'] = appDF['file'].apply(lambda x: Path('NrAudio', x.split('/')[1]))
  appDF['code'] = [[0] * len(TARGET_SPECIES)] * appDF.shape[0]
  appDF.to_csv(Path.cwd().joinpath('data', 'app_test.csv'), header=True, index=False)

def SplitData():
  dataSplitAEC()
  dataSplitAE()
  testAppData()

# -------------
if __name__ == '__main__':
  SplitData()