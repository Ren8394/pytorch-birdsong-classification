import numpy as np
import sys
import torch
import torch.nn as nn

from datetime import datetime
from pathlib import Path
sys.path.append(str(Path.cwd().joinpath('code')))
from torch.utils.data import DataLoader
from tkinter import *
from tkinter.filedialog import askopenfilename
from tqdm import tqdm, trange

from src.network.network import AutoEncoderClassifer
from src.network.dataset import BirdsongDataset
from src.utils.utils import CalculateImbalanceWeight, GetSortedSpeciesCode

# -------------
TARGET_SPECIES = GetSortedSpeciesCode()

# 確定可跑訓練裝置 CPU 或 GPU
if torch.cuda.is_available():
  DEVICE = torch.device('cuda:0')
  torch.backends.cudnn.benchmark = True
else:
  DEVICE = torch.device('cpu')

# 設定 Imbalance Weight
IMBALANCE_WEIGHT = CalculateImbalanceWeight(
  Path.cwd().joinpath('data', 'aec_train.csv'), weightType='ens'
)

# -------------
def train(model, dataloader, optimizer):
  ## Use Automatic Mixed Precision (AMP) to speed up training process
  scaler = torch.cuda.amp.GradScaler()
  model.train()

  ## Loss function
  criterion = nn.BCEWithLogitsLoss(IMBALANCE_WEIGHT).to(DEVICE)   # BCE Loss with Log

  ## Calculate loss, forward and backward process
  trainingLoss = 0.0
  for _, (inputs, labels) in tqdm(
    enumerate(dataloader), total=len(dataloader), bar_format='{l_bar}{bar:32}{r_bar}{bar:-32b}'
  ):
    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
    optimizer.zero_grad()
    with torch.cuda.amp.autocast(enabled=False):
      outputs = model(inputs)
      loss = criterion(outputs, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    trainingLoss += loss.item()

  ## Output loss
  return (trainingLoss / len(dataloader)), model, optimizer

def validate(model, dataloader):
  model.eval()

  ## Loss function
  criterion = nn.BCEWithLogitsLoss(IMBALANCE_WEIGHT).to(DEVICE)
  
  ## Calculate loss
  validationLoss = 0.0
  with torch.no_grad():
    for _, (inputs, labels) in tqdm(
      enumerate(dataloader), total=len(dataloader), bar_format='{l_bar}{bar:32}{r_bar}{bar:-32b}'
    ):
      inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      validationLoss += loss.item()
  
  ## Output loss
  return validationLoss / len(dataloader)

def ExcuteAECTrainingProcess():
  ## Setting
  ## {encoderWeightPath} encoder weight 檔案
  ## {modelFile} model 儲存檔名
  root = Tk()
  root.withdraw()
  encoderWeightPath = askopenfilename(title='Choose The File Of Encoder Weight', initialdir=Path.cwd().joinpath('model'))
  modelFile = Path.cwd().joinpath('model', f'AEClassifer{datetime.now().strftime("%Y%m%d")}.pth')
  root.destroy()

  ## Hyperparameters
  torch.manual_seed(42)
  batchSize = 64
  learningRate = 0.001
  epochs = 250
  earlyStop = 15

  ## Create model and freeze encoder layers 
  model = AutoEncoderClassifer(numberOfClass=len(TARGET_SPECIES)).to(DEVICE)
  model.encoder.load_state_dict(
    torch.load(encoderWeightPath, map_location=DEVICE)
  )
  for param in model.encoder.parameters():
    param.requires_grad = False    # 固定encoder weight不要訓練
  
  ## Optimizer
  optimizer = torch.optim.SGD(model.parameters(), lr=learningRate, momentum=0.9)   # 使用 SGD optimizer

  ## Dataloader
  trainingDataloader = DataLoader(
    BirdsongDataset(Path.cwd().joinpath('data', 'aec_train.csv'), needAugment=True, needLabel=False),
    batch_size=batchSize, shuffle=True, num_workers=6, pin_memory=True
  )
  validationDataloader = DataLoader(
    BirdsongDataset(Path.cwd().joinpath('data', 'aec_validate.csv'), needAugment=False, needLabel=False),
    batch_size=batchSize, shuffle=True, num_workers=6, pin_memory=True
  )

  ## Training
  bestLoss = np.Inf
  earlyCount = 0
  for epoch in trange(epochs, bar_format='{l_bar}{bar:32}{r_bar}{bar:-32b}'):
    trainingLoss, model, optimizer = train(model, trainingDataloader, optimizer)
    validationLoss = validate(model, validationDataloader)

    ### Check loss
    if validationLoss < bestLoss:
      bestLoss = validationLoss
      earlyCount = 0
      torch.save(model.state_dict(), modelFile)
    else:
      earlyCount += 1
      if earlyCount >= earlyStop:
        break

    ### Print results
    print(f"""
      >> [{epoch + 1} / {epochs}] ~~ ~~ AutoEncodeClassifer
      >> {"Best V Loss :":>16} {bestLoss} + [{earlyCount}]
      >> {"Current T Loss :":>16} {trainingLoss:6f}
      >> {"Current V Loss :":>16} {validationLoss:6f}
    """)

# -------------
if __name__ == '__main__':
  ExcuteAECTrainingProcess()