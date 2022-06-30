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
      nn.Conv2d(3, 8, kernel_size=(3,3), stride=2, padding=1),
      nn.BatchNorm2d(8),
      nn.ReLU(True),
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
    self.flatten = nn.Flatten(start_dim=1)
    ## Linear section
    self.encoderLinear = nn.Sequential(
      nn.Linear(8 * 8 * 64, 2048, bias=True),
      nn.ReLU(True),
      nn.Linear(2048, 1024, bias=True)
    )

  def forward(self, inputs):
    x = self.encoderConv(inputs)
    x = self.flatten(x)
    codes = self.encoderLinear(x)
    return codes

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
    self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 8, 8))
    ## Deconvolutional section
    self.decoderConv = nn.Sequential(
      nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
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
    x = self.decoderLinear(codes)
    x = self.unflatten(x)
    outputs = self.decoderConv(x)
    return outputs

# AE
class AutoEncoder(nn.Module):
  def __init__(self):
    super(AutoEncoder, self).__init__()
    self.encoder = Encoder()
    self.decoder = Decoder()

  def forward(self, inputs):
    codes = self.encoder(inputs)
    outputs = self.decoder(codes)
    
    return codes, outputs

# AE + Classifier
class AutoEncodeClassifer(nn.Module):
  def __init__(self, numberOfClass:int) -> None:
    super(AutoEncodeClassifer, self).__init__()
    self.encoder = Encoder()
    self.classifier = nn.Sequential(
      nn.Linear(in_features=1024, out_features=1024, bias=True),
      nn.ReLU(True),
      nn.Linear(in_features=1024, out_features=numberOfClass, bias=True)
    )

  def forward(self, inputs):
    codes = self.encoder(inputs)
    outputs = self.classifier(codes)
    return outputs

# -------------
if __name__ == '__main__':
  ae = AutoEncoder()
  summary(ae, input_size=(16, 3, 128, 128))
  aeClassifier = AutoEncodeClassifer(9)
  summary(aeClassifier, input_size=(16, 3, 128, 128))