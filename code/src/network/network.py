import torch
import torch.nn as nn

from torchinfo import summary

# -------------
# Encoder
class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    ## Convolutional section
    ## Input (batchSize, 3, 128, 128)
    self.encoderConv = nn.Sequential(
      nn.Conv2d(3, 8, kernel_size=(3,3), stride=2, padding=1),    # Convolution
      nn.BatchNorm2d(8),                                          # Batch Normalise
      nn.ReLU(True),                                              # Activation function, ReLu
      nn.Conv2d(8, 16, kernel_size=(3,3), stride=2, padding=1),
      nn.BatchNorm2d(16),
      nn.ReLU(True),
      nn.Conv2d(16, 32, kernel_size=(3,3), stride=2, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(True),
      nn.Conv2d(32, 64, kernel_size=(3,3), stride=2, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(True)
    )
    ## Flatten layer
    self.flatten = nn.Flatten(start_dim=1)                         # Flatten -> 2d to 1d
    ## Linear section
    self.encoderLinear = nn.Sequential(
      nn.Linear(8 * 8 * 64, 2048, bias=True),                      # Fully connect (input=4096, output=2048)
      nn.ReLU(True),                                               # Activation function, ReLu
      nn.Linear(2048, 1024, bias=True)                             # FC
    )

  def forward(self, inputs):
    x = self.encoderConv(inputs)      # Covolutional Layers
    x = self.flatten(x)               # Faltten
    codes = self.encoderLinear(x)     # FC
    return codes                      # Output code

# Decoder
class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    ## Linear section
    self.decoderLinear= nn.Sequential(
      nn.Linear(1024, 2048, bias=True),
      nn.ReLU(True),
      nn.Linear(2048, 8 * 8 * 64, bias=True),
      nn.ReLU(True)
    )
    ## Unflatten layer
    self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 8, 8))         # Unflatten 1d -> 2d
    ## Deconvolutional section
    self.decoderConv = nn.Sequential(
      nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),   # Deconvolution
      nn.BatchNorm2d(32),
      nn.ReLU(True),
      nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
      nn.BatchNorm2d(16),
      nn.ReLU(True),
      nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
      nn.BatchNorm2d(8),
      nn.ReLU(True),
      nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, output_padding=1),
      nn.BatchNorm2d(3)
    )

  def forward(self, codes):
    x = self.decoderLinear(codes)   # FC
    x = self.unflatten(x)           # Unflatten
    outputs = self.decoderConv(x)   # Decovolutional layer
    return outputs                  # output

# AE
class AutoEncoder(nn.Module):
  def __init__(self):
    super(AutoEncoder, self).__init__()
    self.encoder = Encoder()
    self.decoder = Decoder()

  def forward(self, inputs):
    codes = self.encoder(inputs)      # Input -> Encoder -> code
    outputs = self.decoder(codes)     # code -> Decoder -> Output
    
    return codes, outputs             # Get code (for further classifier), and output (self-supervised)

# AE + Classifier
class AutoEncoderClassifer(nn.Module):
  def __init__(self, numberOfClass:int) -> None:    # {numberOfClass} means how many classes do we want to classify
    super(AutoEncoderClassifer, self).__init__()
    self.encoder = Encoder()
    self.classifier = nn.Sequential(
      nn.Linear(in_features=1024, out_features=1024, bias=True),
      nn.ReLU(True),
      nn.Linear(in_features=1024, out_features=numberOfClass, bias=True)
    )

  def forward(self, inputs):
    codes = self.encoder(inputs)        # Input -> Encoder -> code
    outputs = self.classifier(codes)    # code -> FC layer (Classifier) -> outputs
    return outputs

# -------------
if __name__ == '__main__':
  """
    如果執行此檔, 會印出模型架構及各層輸出
    (假設輸入維度 16(batch size)*3(input channel)*128(width)*128(height))
  """
  ae = AutoEncoder()
  summary(ae, input_size=(16, 3, 128, 128))
  aeClassifier = AutoEncoderClassifer(9)
  summary(aeClassifier, input_size=(16, 3, 128, 128))