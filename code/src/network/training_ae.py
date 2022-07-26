import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
import torch
import torch.nn as nn

from datetime import datetime
from pathlib import Path
sys.path.append(str(Path.cwd().joinpath('code')))
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.network.network import AutoEncoder
from src.network.dataset import BirdsongDataset

# -------------
if torch.cuda.is_available():
  DEVICE = torch.device('cuda:1')       # Use second GPU
  torch.backends.cudnn.benchmark = True
else:
  DEVICE = torch.device('cpu')

# -------------
def train(model, dataloader, optimizer):
  ## Use Automatic Mixed Precision (AMP) to speed up training process
  scaler = torch.cuda.amp.GradScaler()
  model.train()

  ## Loss function
  criterion = nn.MSELoss().to(DEVICE)   # MSE Loss

  ## Calculate loss, forward and backward process
  trainingLoss = 0.0
  for _, (inputs, _) in tqdm(
    enumerate(dataloader), total=len(dataloader), bar_format='{l_bar}{bar:32}{r_bar}{bar:-32b}'
  ):
    inputs = inputs.to(DEVICE)
    optimizer.zero_grad()
    with torch.cuda.amp.autocast(enabled=True):
      _, outputs = model(inputs)
      loss = criterion(outputs, inputs)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    trainingLoss += loss.item()
    
  ## Output loss
  return (trainingLoss / len(dataloader)), model, optimizer

def validate(model, dataloader):
  model.eval()

  ## Loss function
  criterion = nn.MSELoss().to(DEVICE)
  
  ## Calculate loss
  validationLoss = 0.0
  with torch.no_grad():
    for _, (inputs, _) in tqdm(
      enumerate(dataloader), total=len(dataloader), bar_format='{l_bar}{bar:32}{r_bar}{bar:-32b}'
    ):
      inputs = inputs.to(DEVICE)
      _, outputs = model(inputs)
      loss = criterion(outputs, inputs)
      validationLoss += loss.item()
  
  ## Output loss
  return validationLoss / len(dataloader)

def ExcuteAETrainingProcess():
  ## Setting
  earlyStatusPath = Path.cwd().joinpath('model', 'AE_checkPoint.tar')
  encoderFile = Path.cwd().joinpath('model', f'AE{datetime.now().strftime("%Y%m%d")}_encoder.pth')

  ## Hyperparameters
  torch.manual_seed(42)
  batchSize = 64
  learningRate = 0.001
  epochs = 100

  ## Create model, optimizer
  ## Load early status
  model = AutoEncoder().to(DEVICE)
  optimizer = torch.optim.SGD(model.parameters(), lr=learningRate, momentum=0.9)      # 使用 SGD optimizer
  if earlyStatusPath.exists():                                                        # 繼續先前進度
    ckPoint = torch.load(earlyStatusPath, map_location=torch.device(DEVICE))          # 
    model.load_state_dict(ckPoint['model_state_dict'])                                # 讀取先前模型 weight
    optimizer.load_state_dict(ckPoint['optimizer_state_dict'])                        # 讀取optimizer 紀錄
    curEpoch = ckPoint['current_epoch']
    bestLoss = ckPoint['best_loss']
  else:
    curEpoch = 0
    bestLoss = np.Inf

  ## Dataloader
  trainingDataloader = DataLoader(
    BirdsongDataset(Path.cwd().joinpath('data', 'ae_train.csv'), needAugment=True, needLabel=True),
    batch_size=batchSize, shuffle=True, num_workers=6, pin_memory=True
  )
  validationDataloader = DataLoader(
    BirdsongDataset(Path.cwd().joinpath('data', 'ae_validate.csv'), needAugment=False, needLabel=True),
    batch_size=batchSize, shuffle=True, num_workers=6, pin_memory=True
  )

  ## Training
  for epoch in tqdm(range(curEpoch, epochs), bar_format='{l_bar}{bar:32}{r_bar}{bar:-32b}'):
    trainingLoss, model, optimizer = train(model, trainingDataloader, optimizer)
    validationLoss = validate(model, validationDataloader)

    ### Check loss
    if validationLoss < bestLoss:
      bestLoss = validationLoss
      torch.save(model.encoder.state_dict(), encoderFile)
    
    ### Save early status
    torch.save({
      'current_epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'best_loss': bestLoss,
    }, earlyStatusPath)

    ### Print results
    print(f"""
      >> [{epoch + 1} / {epochs}]
      >> {"Best Loss :":>16} {bestLoss}
      >> {"Current Train Loss :":>16} {trainingLoss:6f}
      >> {"Current Validate Loss :":>16} {validationLoss:6f}
    """)

# -------------
if __name__ == '__main__':
  ExcuteAETrainingProcess()