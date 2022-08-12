# Pytorch-Birdsong-Classification

## Project Structure

```(python)
< pytorch-birdsong-classification >
  |
  |- birdsongClassification
  |  |
  |  |- birdsongClassifier    // Preprocessing and model functions
  |  |- oneMinuteApplication  // Threshold finding and labeling
  |  |- src                   // Network structure and other utilities
  |  |- audioTransder.ipynb   // For file copy and transfer
  |  |- singleToAuto.ipynb    // Download and sync file from google drive
  |
  |- data   
  |  |- Audio                 // Self-recorded raw audio from automatic record units
  |  |- NrAudio               // Audio which noise is reduced
  |  |- Label                 // Self label using other software
  |  |- OpenSource            // Xeno-canto and eBird data
  |  |- tmp                   // Temporary file for model taining, etc.
  |  |- xxx-dataset.csv       // Manual, auto, single and ae dataset
  |
  |- model                    // Model weight
  |- setting     
  |  |
  |  |- config.ini            // Config for model
  |  |- SPECIES.csv           // For target species selection
  |  |- STATION.csv           // Station list
  |
  |- .gitignore
  |- LICENSE
  |- README.md
  |- requirements.txt 
```

## Installation

### Minicoda

1. Go to [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and download latest Miniconda installer (e.g. _Miniconda3 Windows 64-bit_)

2. Use installer to install miniconda

   * In advanced options, check all boxes about **Add to my PATH** and **Register default Python**

3. Open command line interface (type `cmd` in search), and type `conda info` to see whether the conda environment install correctly

### VSCode

1. Go to [VSCode](https://code.visualstudio.com/download) and download user installer

2. Use installer to install VSCode

   * Checking boxes for **add PATH**, **Open with code**, ... is recommanded

3. Open VSCode and search for Python extension, or you can use this [link](https://marketplace.visualstudio.com/items?itemName=ms-python.python) to download extension. Also, if you want to run jupyter notebook in VSCode, installing [jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) is recommanded.

### Setup environment

#### Windows Platform

If you are in Windows OS, just run `conda env create -f environment.yml` in the _Pytorch-Birdsong-Classification_ folder. All necessary packages will be installed in **pytorch_env** environment.

#### Other platforms

If you are in other platforms such as MacOS or Linux. Create a conda environment and install necessary packages with the following command.  

1. `conda env create -n pytorch_env`
2. `conda activate pytorch_env`
3. install packages

   * absl-py `pip install absl-py`  
   * jupyter `conda install -c ananconda jupyter`
   * noisereduce `pip install noisereduce`  
   * librosa `conda install -c conda-forge librosa`  
   * pydub `pip install pydub`  
   * seaborn `conda install seaborn`  
   * soundfile `pip install SoundFile`  
   * scipy `conda install -c anaconda scipy`  
   * sklearn `conda install -c intel scikit-learn`  
   * tqdm `conda install -c conda-forge tqdm`  
   * torch `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`  
   * torchinfo `conda install -c conda-forge torchinfo`  
