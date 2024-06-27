
import numpy as np
import torch
from torch.utils.data import IterableDataset
import torchaudio as ta
from torchaudio.compliance.kaldi import fbank

NON_LABELED_AUDIO_DIR = "~/Desktop/Audio_data/pretrain_Audio"
LABELED_AUDIO_DIR = "~/Desktop/Audio_data/finetune_Audio"

AUDIO_LENGTH = 60 * 1000    # ms
WINDOW_SIZE = 3000          # ms
HOP_LENGTH = 10             # ms

class TWBird(IterableDataset):
    def __init__(self, src_file, labeled=False):
        super(TWBird, self).__init__()
        
        self.labeled = labeled

        self.file_paths = np.loadtxt(src_file, dtype=str)
        
        self.iter_count = 0


    def __len__(self):
        return int(len(self.file_paths) * (AUDIO_LENGTH - WINDOW_SIZE) // HOP_LENGTH)

    def __iter__(self):
        waveform, sr = ta.load(self.file_paths[self.iter_count])

        if not self.labeled:
            pass

    def augment(self, waveform):
        pass



if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = TWBird(src_file="./data/pretrain/test.txt")
    # dataset = TWBird(src_file="./data/finetune/train.txt")

    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    # for _ in dataloader:
    #     break