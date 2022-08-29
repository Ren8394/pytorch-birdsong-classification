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

1. Execute `conda env create -f environment.yml` in the _Pytorch-Birdsong-Classification_ folder.
   * It will create **pytorch_env** environment and install conda packages in the environment.
2. Execute `conda activate pytorch_env` getting into the **pytorch_env** environment
