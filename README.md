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

4. In CLI, type `conda create -n {name} python={version}` to create conda enviornment

   * We usually create another environment to saperate environment from _base_
   * In this doc, execute `conda create -n py39 python=3.9`, which means we create an environment named _py39_ and install python _version 3.9_

5. After successfuly creating _py39_ enviornment, execute `conda activate py39` to activate enviornment. You will see that the word in parentheses changed from _base_ to _py39_

### VSCode

1. Go to [VSCode](https://code.visualstudio.com/download) and download user installer

2. Use installer to install VSCode

   * Checking boxes for add PATH, Open with code, ... is recommanded

3. Open VSCode and search for Python extension, or you can use this [link](https://marketplace.visualstudio.com/items?itemName=ms-python.python) to download extension

4. While extension istallation is finish, VSCode will ask for selecting interpreter.
Select _py39_ which we used conda to create before

### Necessary Packages Of Our Project

#### Use requirements.txt

  1. In project folder, open Windows CLI or VSCode (Ctrl + ` to open CLI in VSCode).

     * If set interpreter correctly in VSCode part, VSCode will activate _py39_ environment automatically
     * If use Windows CLI, please remember activating _py39_ environment first

  2. Execute `pip install -r requirements.txt` to install necessary packages

  3. Execute `conda list` to check whether the packages install properly or not

#### Use every offical instruction

absl-py `pip install absl-py`  
noisereduce `pip install noisereduce`  
librosa `conda install -c conda-forge librosa`  
pydub `pip install pydub`  
seaborn `conda install seaborn`  
soundfile `pip install SoundFile`  
scipy `conda install -c anaconda scipy`  
sklearn `conda install -c intel scikit-learn`  
tqdm `conda install -c conda-forge tqdm`  
torch `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`  
torchinfo `conda install -c conda-forge torchinfo`  
