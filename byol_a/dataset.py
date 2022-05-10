"""BYOL for Audio: Dataset class definition."""

from .common import (random, np, torch, F, torchaudio, AF, AT, Dataset)
import librosa
import os

class MelSpectrogramLibrosa:
    """Mel spectrogram using librosa."""
    def __init__(self, fs=16000, n_fft=1024, shift=160, n_mels=64, fmin=60, fmax=7800):
        self.fs, self.n_fft, self.shift, self.n_mels, self.fmin, self.fmax = fs, n_fft, shift, n_mels, fmin, fmax
        self.mfb = librosa.filters.mel(sr=fs, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

    def __call__(self, audio):
        X = librosa.stft(np.array(audio), n_fft=self.n_fft, hop_length=self.shift)
        ret = torch.tensor(np.matmul(self.mfb, np.abs(X)**2 + np.finfo(float).eps))
        # print("mel_:", ret.size())
        return ret
    
class Precompute_Mel:
    """Mel spectrogram using librosa."""
    def __init__(self, n_mel, segment_frames):
        self.segment_length = segment_frames
        self.n_mel = n_mel
        
    def __call__(self, path):
        # print(path)
        path = str(path).replace(".wav","_logmel.npy")
        # path = str(path).replace(".wav","_logmel.npy")
        _buffer = torch.zeros((self.n_mel,self.segment_length))
        mat = torch.tensor(np.load(path).T)
        
        if(mat.size(1)-self.segment_length <= 0): start = 0
        else: start = int(np.random.uniform(low=0, high=mat.size(1)-self.segment_length))
        
        _buffer[:,:int(min(self.segment_length,mat.size(1)))] = mat[:,start:start + self.segment_length]
        # print(mat.size())
        return _buffer

def draw(array, name="temp.png"):
    import matplotlib.pyplot as plt
    plt.imshow(array.numpy()[0], aspect="auto")
    plt.savefig(name)
    plt.close()


class WaveInLMSOutDataset(Dataset):
    """Wave in, log-mel spectrogram out, dataset class.

    Choosing librosa or torchaudio:
        librosa: Stable but slower.
        torchaudio: Faster but cannot reproduce the exact performance of pretrained weight,
            which might be caused by the difference with librosa. Librosa was used in the pretraining.

    Args:
        cfg: Configuration settings.
        audio_files: List of audio file pathnames.
        labels: List of labels corresponding to the audio files.
        tfms: Transforms (augmentations), callable.
        use_librosa: True if using librosa for converting audio to log-mel spectrogram (LMS).
    """

    def __init__(self, cfg, audio_files, labels, tfms, use_librosa=False):
        # argment check
        assert (labels is None) or (len(audio_files) == len(labels)), 'The number of audio files and labels has to be the same.'
        super().__init__()

        # initializations
        self.cfg = cfg
        self.files = audio_files
        self.labels = labels
        self.tfms = tfms
        self.unit_length = int(cfg.unit_sec * cfg.sample_rate)
        temp_files = []
        for each in self.files:
            if(os.path.exists(each)): temp_files.append(each)
        print("Originally we have %s files, but there is only % exist" % (len(self.files), len(temp_files)))
        self.files = temp_files
        ###################################################################################
        # self.to_melspecgram = MelSpectrogramLibrosa(
        #     fs=cfg.sample_rate,
        #     n_fft=cfg.n_fft,
        #     shift=cfg.hop_length,
        #     n_mels=cfg.n_mels,
        #     fmin=cfg.f_min,
        #     fmax=cfg.f_max,
        # ) if use_librosa else AT.MelSpectrogram(
        #     sample_rate=cfg.sample_rate,
        #     n_fft=cfg.n_fft,
        #     win_length=cfg.win_length,
        #     hop_length=cfg.hop_length,
        #     n_mels=cfg.n_mels,
        #     f_min=cfg.f_min,
        #     f_max=cfg.f_max,
        #     power=2,
        # )
        ###################################################################################
        self.to_melspecgram2=Precompute_Mel(n_mel=cfg.n_mels, segment_frames = int(self.unit_length/cfg.hop_length)+1)
        ###################################################################################
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        ###################################################################################
        # # # load single channel .wav audio
        # wav, sr = torchaudio.load(self.files[idx])
        # assert sr == self.cfg.sample_rate, f'Convert .wav files to {self.cfg.sample_rate} Hz. {self.files[idx]} has {sr} Hz.'
        # assert wav.shape[0] == 1, f'Convert .wav files to single channel audio, {self.files[idx]} has {wav.shape[0]} channels.'
        # wav = wav[0] # (1, length) -> (length,)

        # # zero padding to both ends
        # length_adj = self.unit_length - len(wav)
        # if length_adj > 0:
        #     half_adj = length_adj // 2
        #     wav = F.pad(wav, (half_adj, length_adj - half_adj))

        # # random crop unit length wave
        # length_adj = len(wav) - self.unit_length
        # start = random.randint(0, length_adj) if length_adj > 0 else 0
        # wav = wav[start:start + self.unit_length]
        # print(wav.size(), start, self.unit_length)
        
        # # # to log mel spectrogram -> (1, n_mels, time)
        # lms = (self.to_melspecgram(wav) + torch.finfo().eps).log().unsqueeze(0)
        # # draw(lms,"lms1.png")
        ###################################################################################
        if(idx >= len(self.files)): idx=0
        lms = (self.to_melspecgram2(self.files[idx]) + torch.finfo().eps).unsqueeze(0)
        # draw(lms2,"lms2.png")
        ###################################################################################
        # transform (augment)
        if self.tfms:
            lms = self.tfms(lms)

        if self.labels is not None:
            return lms, torch.tensor(self.labels[idx])
        
        return lms

