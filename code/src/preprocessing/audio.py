import librosa
import noisereduce as nr
import pandas as pd
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# -------------
def noiseReduction(filePaths:Path):
  """ 降噪
    1. 讀取未降噪檔案
    2. 標準化音檔 (使用 librosa > librosa.util.normalize)
    3. 降噪 (使用 noisereduce package)
    4. 匯出降噪後音檔於 (pwd/data/rwa/NrAudio, 格式PCM_24, wav)
    5. 刪除未降噪檔案
  """
  for filePath in tqdm(filePaths, bar_format='{l_bar}{bar:32}{r_bar}{bar:-32b}'):
    audio, sr = librosa.load(str(filePath), sr=None)
    audio = audio.T
    audio = librosa.util.normalize(audio)
    audio = nr.reduce_noise(y=audio, sr=sr, n_jobs=-1, use_tqdm=True)
    ## Save noise-reduced audio in wav format extension 
    sf.write(
      Path.cwd().joinpath('data', 'raw', 'NrAudio', f'{filePath.stem}.wav'),
      audio, sr, 'PCM_24'
    )
    Path.unlink(filePath)

def concatAudio(filePaths:Path, labeledFilePaths:Path):
  """ 音訊 metadata (實際未應用，未確認功能)
    依照檔案取出相關資料
    file: 降噪音檔路徑
    labeled: 有無人工標記
    station: 測站名稱
    data: 錄音日期
    record time: 錄音時間
  """ 
  df = pd.DataFrame(columns=['file', 'labeled', 'station', 'date', 'record time'])
  for filePath in tqdm(filePaths, bar_format='{l_bar}{bar:32}{r_bar}{bar:-32b}'):
    labelFunc = lambda x, y: True if x in y else False # Check the file labled or not
    df = pd.concat(
      [df, pd.DataFrame({
        'file': Path('NrAudio', filePath.name),
        'labeled': labelFunc(filePath, labeledFilePaths),
        'station': str(filePath.stem).split('_')[0],
        'date': datetime.strptime((filePath.stem).split('_')[1], '%Y%m%d').date(),
        'record time': datetime.strptime((filePath.stem).split('_')[2], '%H%M%S').time().strftime('%H:%M')
      }, index=[0])], ignore_index=True
    )
  return df

def ReduceAudioNoise():
  """
    1. 將 {sAudioPaths} 路徑下所有錄音檔進行降噪處理於 NrAudio 存成相同檔名的降噪檔案
    2. 將已經人工標籤過的檔案 {sLabeledNrAudioPaths} 取出製作成檔名為 AUDIO.csv metadata
    3. 將 {oLabelAudioPaths} 路徑下所有錄音檔進行降噪處理於 NrAudio 存成相同檔名的降噪檔案
    // {sAudioPaths} 未降噪私人錄音路徑
    // {sLabeledNrAudioPaths} 有標記的已降噪私人錄音檔案路徑
    // {oLabelAudioPaths} 有標記的OpenSource檔案路徑
  """
  ## Self-record audio noise reduction
  sAudioPaths = sorted(Path.cwd().joinpath('data', 'raw', 'Audio').glob('*.wav'))
  noiseReduction(sAudioPaths)

  sLabeledNrAudioPaths = list(map(
    lambda x: sorted(Path.cwd().joinpath('data', 'raw', 'NrAudio').glob(f'*{x.stem}*'))[0],
    sorted(Path.cwd().joinpath('data', 'raw', 'Label').glob('*.txt'))
  ))
  sNrAudioPaths = sorted(Path.cwd().joinpath('data', 'raw', 'NrAudio').glob('*.wav'))
  audioDF = concatAudio(sNrAudioPaths, sLabeledNrAudioPaths)
  audioDF.to_csv(Path.cwd().joinpath('data', 'AUDIO.csv'), header=True, index=False)

  oLabeledAudioPaths = list(map(
    lambda x: sorted(Path.cwd().joinpath('data', 'raw', 'Opensource').glob(f'*{x.stem}*'))[0],
    sorted(Path.cwd().joinpath('data', 'raw', 'Opensource').glob('*.txt'))
  ))
  noiseReduction(oLabeledAudioPaths)

# -------------
if __name__ == '__main__':
  ReduceAudioNoise()
