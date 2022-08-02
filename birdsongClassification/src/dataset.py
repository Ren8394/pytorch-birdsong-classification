import ast
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
# -------------
AUDIOCLIP_LEN = 2.0   # 每一個音檔轉成圖片前固定長度

# -------------
def audioAugment(audio, samplingRate, probability=0.2):
  """ Data Augmentation
    每一種 augment 機率 {probability, default 20%}
    augment 方法: Time stretch, Pitch shifting (每種內含4種參數, 25%)
  """
  ## Time stretch  {0.81, 0.93, 1.07, 1.23}
  if np.random.random(size=None) <= probability:
    choice = np.random.randint(low=1, high=5)
    if choice == 1:
      audio = librosa.effects.time_stretch(audio, rate=0.81)
    elif choice == 2:
      audio = librosa.effects.time_stretch(audio, rate=0.93)
    elif choice == 3:
      audio = librosa.effects.time_stretch(audio, rate=1.07)
    else:
      audio = librosa.effects.time_stretch(audio, rate=1.23)

  ## Pitch shifting {−2, −1, 1, 2}  
  if np.random.random(size=None) <= probability:
    choice = np.random.randint(low=1, high=5)
    if choice == 1:
      audio = librosa.effects.pitch_shift(audio, sr=samplingRate, n_steps=-2)
    elif choice == 2:
      audio = librosa.effects.pitch_shift(audio, sr=samplingRate, n_steps=-1)
    elif choice == 3:
      audio = librosa.effects.pitch_shift(audio, sr=samplingRate, n_steps=1)
    else:
      audio = librosa.effects.pitch_shift(audio, sr=samplingRate, n_steps=2)

  return audio

def generatePCNEMelSpec(audio, samplingRate, window):
  """
    將音檔轉換成 Mel-時頻圖, 並使用PCEN技術標準化
    Mel 轉換視窗大小 {window = (視窗大小, 滑移距離)}
    ----
    Rethinking CNN models for audio classification (2020)
    Per-channel energy normalization: Why and how (2018)
    Chirping up the right tree: Incorporating biological taxonomies into deep bioacoustic classifiers (2020)
    ----
  """
  pcenTime = 0.06
  pcenGain = 0.8
  pcenBias = 10
  pcenPower = 0.25
  ## window = (fft window, hop length)
  melSpec = librosa.feature.melspectrogram(
    y=audio, sr=samplingRate,
    n_fft=int(window[0] * samplingRate), hop_length=int(window[1] * samplingRate),
    n_mels=128, fmin=1000, fmax=10000
  )
  image = librosa.pcen(
    melSpec * (2**31), sr=samplingRate,
    time_constant=pcenTime, gain=pcenGain, bias=pcenBias, power=pcenPower
  )
  return image

# -------------
class BirdsongDataset(torch.utils.data.Dataset):
  """
    資料集  
  """
  def __init__(self, filePath, needAugment=True, needLabel=False):
    self.dataDF = pd.read_csv(filePath, header=0)   # 資料集路徑
    self.needAugment = needAugment                  # 是否需要 data augment
    self.needLabel = needLabel                      # 是否需要 label 輸出

  def __len__(self):
    return len(self.dataDF)

  def __getitem__(self, index):
    """
      1. 將音訊進行 Augment
      2. 將音檔用複製模式(wrap)為 {AUDIOCLIP_LEN} 長度
      3. 使用3種不同大小及滑移距離的視窗, 將音訊轉PCEN Mel-時頻圖
      4. 將3種設定產出的圖片以插值法固定大小為 128 pixel * 128 pixel
      5. 將3張圖片疊起成1次輸入
        1. 如果不需要Label {needLabel = false} 輸出零向量label
        2. 如需要{needLabel = true}, 輸出One-hot label
    """    
    ## Audio source
    folderPath = Path.cwd().parent.parent.joinpath('data')
    audioFilePath = str(folderPath.joinpath(self.dataDF.loc[index, 'file']))
    audio, sr = librosa.load(audioFilePath, sr=None)
    audio = audio.T
    audio = audio[int(float(self.dataDF.loc[index, 'start time'])*sr):int(float(self.dataDF.loc[index, 'end time'])*sr)]

    ## DataAugument
    if self.needAugment:
      audio = audioAugment(audio, sr, probability=0.2)

    ## Mel-spectrogram and PCEN
    audio = librosa.util.fix_length(audio, size=int(AUDIOCLIP_LEN * sr), mode='wrap')
    imgS = F.interpolate(torch.as_tensor(generatePCNEMelSpec(audio, sr, window=(0.025, 0.010)))[None, None, :, :], (128, 128)).squeeze(dim=0)
    imgM = F.interpolate(torch.as_tensor(generatePCNEMelSpec(audio, sr, window=(0.050, 0.025)))[None, None, :, :], (128, 128)).squeeze(dim=0)
    imgL = F.interpolate(torch.as_tensor(generatePCNEMelSpec(audio, sr, window=(0.100, 0.050)))[None, None, :, :], (128, 128)).squeeze(dim=0)
    inputImage = torch.cat((imgS, imgM, imgL), dim=0).float()

    ## The label of autoencoder is itself
    if self.needLabel:
      onehotLabel = torch.FloatTensor(ast.literal_eval(self.dataDF.loc[index, 'onehot']))
      return inputImage, onehotLabel
    else:
      return inputImage, torch.zeros(0)

# -------------
if __name__ == '__main__':
  pass