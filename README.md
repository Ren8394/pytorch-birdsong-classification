# Pytorch-Birdsong-Classification

## Project Structure

```(python)
< pytorch-birdsong-classification >
  |
  |- assets                                // Image
  |- qml                                   // Qml file for GUI (*TODO)
  |- model                                 // Model Weight
  |- code           
  |    |- app.py                            // Execute for sigle file testing
  |    |- src
  |        |- network 
  |        |   |- dataset.py                // Birdsong dataset
  |        |   |- network.py                // Network structure
  |        |   |- testing_overall.py        // Test model performace 
  |        |   |- testing_single.py         // Test single audio file
  |        |   |- training_ae.py            // Training AutoEncoder (AE)
  |        |   |- training_aec.py           // Training AutoEncoder + Classifier (AEC)
  |        |
  |        |- preprocessing
  |        |   |- audio.py                  // Reduce noise
  |        |   |- data_split.py             // Split data
  |        |   |- label.py                  // Generate label from self-labeled data
  |        |
  |        |- utils
  |        |   |- auto_label.py             // Label non-labeled automatically
  |        |   |- result_visualisation.py   // Visualise and correct predict using CMD interface
  |        |   |- threshold_app.py          // Get proper threshold for application usage automatically
  |        |   |- utils.py                  // Many utility functions
  |
  |- data
      |- raw
      |    |- Audio                         // Raw audio
      |    |- Label                         // Manual labels
      |    |- NrAudio                       // Reduced noise audio
      |    |- Opensource                    // Xeno-Canto and eBird audio and label
      |
      |- SPECIES.csv                                        // Species info
      |- STATION.csv                                        // Station info
      |
      |- audio.csv                                          // Audio metadata
      |
      |- label_ae.csv                                       // Autoencoder dataset
      |- label_auto.csv                                     // Auto label dataset (window)
      |- label_seg.csv                                      // Manual lable dataset (segment algorithm)
      |- label_single.csv                                   // Single test label dataset
      |
      |- ae_train.csv, ae_validate.csv                      // Splited autoencoder set
      |- aec_train.csv, aec_validate.csv, aec_test.csv      // Splited autoencoder Ccassifier set
      |- app_test.csv, test_app.csv                         // Application test and its output result
      |- res_single.csv, tempTest.csv                       // Temporary file

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

Use requirements.txt

  1. In project folder, open Windows CLI or VSCode (Ctrl + ` to open CLI in VSCode).

     * If set interpreter correctly in VSCode part, VSCode will activate _py39_ environment automatically
     * If use Windows CLI, please remember activating _py39_ environment first

  2. Execute `pip install -r requirements.txt` to install necessary packages

  3. Execute `conda list` to check whether the packages install properly or not

Use every offical instruction

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

<!-- scipy `pip install numpy scipy matplotlib ipython jupyter pandas sympy nose` -->  
<!-- librosa `pip install librosa` --> 