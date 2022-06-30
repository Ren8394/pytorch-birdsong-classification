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

def concatAudio(filePaths:Path, labeledFilePaths:Path):
  df = pd.DataFrame(columns=['file', 'labeled', 'station', 'date', 'record time'])
  for filePath in tqdm(filePaths, bar_format='{l_bar}{bar:32}{r_bar}{bar:-32b}'):
    labelFunc = lambda x, y: True if x in y else False # Check the file labled or not
    df = pd.concat(
      [df, pd.DataFrame({
        'file': Path('Audio', filePath.name),
        'labeled': labelFunc(filePath, labeledFilePaths),
        'station': str(filePath.stem).split('_')[0],
        'date': datetime.strptime((filePath.stem).split('_')[1], '%Y%m%d').date(),
        'record time': datetime.strptime((filePath.stem).split('_')[2], '%H%M%S').time().strftime('%H:%M')
      }, index=[0])], ignore_index=True
    )
  return df

def ReduceAudioNoise():
  ## Labeled audio > noise-reduced audio
  ## Same as labeled open-source audio
  sLabeledAudioPaths = list(map(
    lambda x: sorted(Path.cwd().joinpath('data', 'raw', 'Audio').glob(f'*{x.stem}*'))[0],
    sorted(Path.cwd().joinpath('data', 'raw', 'Label').glob('*.txt'))
  ))
  oLabeledAudioPaths = list(map(
    lambda x: sorted(Path.cwd().joinpath('data', 'raw', 'Opensource').glob(f'*{x.stem}*'))[0],
    sorted(Path.cwd().joinpath('data', 'raw', 'Opensource').glob('*.txt'))
  ))
  noiseReduction(sLabeledAudioPaths + oLabeledAudioPaths)

  ## Self-record audio
  sAudioPaths = sorted(Path.cwd().joinpath('data', 'raw', 'Audio').glob('*.wav'))
  audioDF = concatAudio(sAudioPaths, sLabeledAudioPaths)
  audioDF.to_csv(Path.cwd().joinpath('data', 'AUDIO.csv'), header=True, index=False)

# -------------
if __name__ == '__main__':
  ReduceAudioNoise()
