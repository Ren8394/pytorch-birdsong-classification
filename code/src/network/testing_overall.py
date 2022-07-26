import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pandas as pd
import sys
import torch
import torch.nn as nn

from pathlib import Path
sys.path.append(str(Path.cwd().joinpath('code')))
from sklearn.metrics import classification_report, precision_score, recall_score, fbeta_score
from torch.utils.data import DataLoader
from tkinter import *
from tkinter.filedialog import askopenfilename
from tqdm import tqdm

from src.network.network import AutoEncoderClassifer
from src.network.dataset import BirdsongDataset
from src.utils.utils import GetSortedSpeciesCode

# -------------
TARGET_SPECIES = GetSortedSpeciesCode()

if torch.cuda.is_available():
  DEVICE = torch.device('cuda:0')       # Use first GPU
  torch.backends.cudnn.benchmark = True
else:
  DEVICE = torch.device('cpu')

# -------------
def test(model, dataloader):
  model.eval()

  predicts, actuals = [], []
  with torch.no_grad():
    for _, (inputs, labels) in tqdm(
      enumerate(dataloader), total=len(dataloader), bar_format='{l_bar}{bar:32}{r_bar}{bar:-32b}'
    ):
      inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
      outputs = nn.functional.sigmoid(model(inputs))
      predicts.extend(outputs.cpu().numpy())
      actuals.extend(labels.cpu().numpy())
  return predicts, actuals

def resultsVisualisation(predicts, actuals):
  ## Get statistics results
  res = {}
  trueLabels = np.array(np.reshape(actuals, (-1, len(TARGET_SPECIES))), dtype=int)
  for thres in np.arange(0, 0.95, 0.01):
    predLabels = np.array(
      np.reshape(predicts, (-1, len(TARGET_SPECIES))) >= thres, dtype=int
    )
    res[thres] = classification_report(
      y_true=trueLabels, y_pred=predLabels,
      target_names=TARGET_SPECIES, zero_division=0, output_dict=True
    )

  ## F1, Precision, and Recall
  f1 = dict((specie, []) for specie in TARGET_SPECIES)
  precision = dict((specie, []) for specie in TARGET_SPECIES)
  recall = dict((specie, []) for specie in TARGET_SPECIES)
  for thres, vs in res.items():
    for k, v in vs.items():
      if k in TARGET_SPECIES:
        f1[k].append(v['f1-score'])
        precision[k].append(v['precision'])
        recall[k].append(v['recall'])


  ## Plot
  thresList = list(np.around(np.arange(0, 0.95, 0.01), decimals=2))
  fig, axs = plt.subplots(
    int(np.sqrt(len(TARGET_SPECIES))), int(np.sqrt(len(TARGET_SPECIES))), 
    figsize=(12, 12), tight_layout=True
  )
  axs = axs.flatten()
  for k, v in precision.items():
    axs[TARGET_SPECIES.index(k)].set_ylim(0, 1)
    axs[TARGET_SPECIES.index(k)].set_xlim(0, 1)
    axs[TARGET_SPECIES.index(k)].plot(thresList, v, 'r-')     # 紅線 > precision
    axs[TARGET_SPECIES.index(k)].set_title(k)
  for k, v in recall.items():
    axs[TARGET_SPECIES.index(k)].set_ylim(0, 1)
    axs[TARGET_SPECIES.index(k)].set_xlim(0, 1)
    axs[TARGET_SPECIES.index(k)].plot(thresList, v, 'k-')     # 黑線 > recall
    axs[TARGET_SPECIES.index(k)].set_title(k)
  for k, v in f1.items():
    axs[TARGET_SPECIES.index(k)].set_ylim(0, 1)
    axs[TARGET_SPECIES.index(k)].set_xlim(0, 1)
    axs[TARGET_SPECIES.index(k)].plot(thresList, v, 'b-')     # 藍線 > F1
    axs[TARGET_SPECIES.index(k)].set_title(f'{k} \nMax F1 ({np.max(v):.2f}) Threshold: {thresList[np.argmax(v)]}')
  plt.savefig(Path.cwd().joinpath('report.png'))

def showStatisticResults(predicts, actuals):
  ## Build table frame
  staticDFIndex = pd.MultiIndex.from_product(
    [TARGET_SPECIES, ['precision', 'recall', 'f0.5', 'f1', 'f2']]
  )
  thresList = np.around(np.arange(0, 1, 0.01), decimals=2)
  staticDF = pd.DataFrame(columns=thresList, index=staticDFIndex)
  
  ## Fill in table
  trueLabels = np.array(np.reshape(actuals, (-1, len(TARGET_SPECIES))), dtype=int)
  for thres in thresList:
    predLabels = np.array(
      np.reshape(predicts, (-1, len(TARGET_SPECIES))) >= thres, dtype=int
    )
    for i, sp in enumerate(TARGET_SPECIES):
      staticDF.loc[(sp, 'precision'), thres] = np.round(
        precision_score(
          y_pred=predLabels[:, i], y_true=trueLabels[:, i], zero_division=0
        ), decimals=4
      )
      staticDF.loc[(sp, 'recall'), thres] = np.round(
        recall_score(
          y_pred=predLabels[:, i], y_true=trueLabels[:, i], zero_division=0
        ), decimals=4
      )
      staticDF.loc[(sp, 'f0.5'), thres] = np.round(
        fbeta_score(
          y_pred=predLabels[:, i], y_true=trueLabels[:, i], beta=0.5, zero_division=0,
        ),
        decimals=4
      )
      staticDF.loc[(sp, 'f1'), thres] = np.round(
        fbeta_score(
          y_pred=predLabels[:, i], y_true=trueLabels[:, i], beta=1.0, zero_division=0,
        ),
        decimals=4
      )
      staticDF.loc[(sp, 'f2'), thres] = np.round(
        fbeta_score(
          y_pred=predLabels[:, i], y_true=trueLabels[:, i], beta=2, zero_division=0,
        ),
        decimals=4
      )

  ## Save results
  staticDF.fillna(0, inplace=True)
  staticDF.T.to_csv(Path.cwd().joinpath('data', 'test_overall.csv'), header=True, index=True)

def ExcuteOverallTestingProcess():
  ## Setting
  ## {WeightPath} 讀取 model weight
  root = Tk()
  root.withdraw()
  WeightPath = askopenfilename(title='Choose The File Of Model Weight', initialdir=Path.cwd().joinpath('model'))
  root.destroy()

  ## model
  model = AutoEncoderClassifer(numberOfClass=len(TARGET_SPECIES)).to(DEVICE)
  model.load_state_dict(
    torch.load(WeightPath, map_location=torch.device(DEVICE))
  )

  ## Dataloader
  testingDataloader = DataLoader(
    BirdsongDataset(Path.cwd().joinpath('data', 'aec_test.csv'), needAugment=False, needLabel=False),
    batch_size=8, shuffle=False, num_workers=6, pin_memory=True
  )
  appTestingDataLoader = DataLoader(
    BirdsongDataset(Path.cwd().joinpath('data', 'app_test.csv'), needAugment=False, needLabel=False),
    batch_size=8, shuffle=False, num_workers=6, pin_memory=True
  )

  ## Test for model
  predicts, actuals = test(model, testingDataloader)
  ## Save results output
  showStatisticResults(predicts, actuals)
  ## Classification reports
  resultsVisualisation(predicts, actuals)


  ## Test for applicatation
  predicts, actuals = test(model, appTestingDataLoader)
  ## Save Predict Probability
  predicts = np.array(np.reshape(predicts, (-1, len(TARGET_SPECIES))))
  tempDF = pd.read_csv(Path.cwd().joinpath('data', 'app_test.csv'), header=0)
  resDF = pd.concat(
    [
      tempDF[['file', 'start time', 'end time']], 
      pd.DataFrame(predicts, columns=[f'{sp}(P)' for sp in TARGET_SPECIES])
    ], axis=1, ignore_index=False
  )
  if Path.cwd().joinpath('data', 'test_app.csv').exists():
    testDF = pd.read_csv(Path.cwd().joinpath('data', 'test_app.csv'), header=0)
    testDF = pd.concat([testDF, resDF], ignore_index=True)
    testDF.drop_duplicates(subset=['file', 'start time', 'end time'], inplace=True)
    testDF.sort_values(by=['file', 'start time', 'end time'], inplace=True)
    testDF.to_csv(Path.cwd().joinpath('data', 'test_app.csv'), header=True, index=False)
  else:
    resDF.to_csv(Path.cwd().joinpath('data', 'test_app.csv'), header=True, index=False)

# -------------
if __name__ == '__main__':
  ExcuteOverallTestingProcess()