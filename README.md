# Pytorch-Birdsong-Classification

## Project Structure

```(python)
< pytorch-birdsong-classification >
  |
  |- assets                                // Image
  |- qml                                   // Qml file for GUI (*TODO)
  |- code           
      |- app.py                            // Execute for sigle file testing
      |- src
          |- network 
          |   |- dataset.py                // Birdsong dataset
          |   |- network.py                // Network structure
          |   |- testing_overall.py        // Test model performace 
          |   |- testing_single.py         // Test single audio file
          |   |- training_ae.py            // Training AutoEncoder (AE)
          |   |- training_aec.py           // Training AutoEncoder + Classifier (AEC)
          |
          |- preprocessing
          |   |- audio.py                  // Reduce noise
          |   |- data_split.py             // Split data
          |   |- label.py                  // Generate label from self-labeled data
          |
          |- utils
          |   |- auto_label.py             // Label non-labeled automatically
          |   |- result_visualisation.py   // Visualise and correct predict using CMD interface
          |   |- threshold_app.py          // Get proper threshold for application usage automatically
          |   |- utils.py                  // Many utility functions
```
