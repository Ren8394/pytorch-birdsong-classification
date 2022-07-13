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
