import os
import tqdm
import h5py
import librosa
import numpy as np
import pandas as pd
from seld.utils.features_extractor import afextractor

class Preprocessor():
    def __init__(self, cfg):
        self.data_rootdir = cfg['data rootdir']
        self.dataset = cfg['dataset']['name']

        self.data_dir = os.path.join(self.data_rootdir, self.dataset, 'data')

    def extract_features(self, cfg):
        features_foldname = "{}_sr{}_nfft{}_hoplen{}_nmels{}".format(cfg['features']['type'], 
                                cfg['features']['sample rate'], cfg['features']['n_fft'], cfg['features']['hop_len'], cfg['features']['n_mels'])
        if "SALSA" in cfg['features']['type']:
            features_foldname += "_SALSAwinsize{}".format(cfg['features']['SALSA win size']) 

        features_dir = os.path.join(self.data_rootdir, self.dataset, '_h5', features_foldname)
        assert os.path.isdir(features_dir) == False, f'{features_dir} is already exists'   

        total = []

        if cfg['audio format'] == 'foa':
            audio_sets = ['foa_dev', 'foa_eval']
        if cfg['audio format'] == 'mic':
            audio_sets = ['mic_dev', 'mic_eval']
        extractor = afextractor(cfg)

        for audio_set in audio_sets:
            os.makedirs(os.path.join(features_dir, audio_set))
            wav_fns = [fn for fn in os.listdir(os.path.join(self.data_dir, audio_set)) if ".wav" in fn]

            for wav_fn in wav_fns:
                waveform, _ = librosa.load(os.path.join(self.data_dir, audio_set, wav_fn), sr=cfg['features']['sample rate'], mono=False, dtype=np.float32)
                features = extractor.extract_features(waveform)
                total.append(features)
                h5_fn = wav_fn.replace('wav', 'h5')
                with h5py.File(os.path.join(features_dir, audio_set, h5_fn), 'w') as hf:
                    hf.create_dataset(name='features', data=features, dtype=np.float32)

        total = np.stack(total, axis=0)
        mean = np.mean(total, axis=(0,4), keepdims=True)
        std = np.std(total, axis=(0,4), keepdims=True)

        with h5py.File(os.path.join(features_dir, 'scalar.h5'), 'w') as hf:
            hf.create_dataset(name='mean', data=mean, dtype=np.float32)        
            hf.create_dataset(name='std', data=std, dtype=np.float32)        

    def extract_metadata(self):
        if self.dataset != 'TNSSE2020':
            return

        metadata_sets = ['metadata_dev', 'metadata_eval']
        for metadata_set in metadata_sets:
            metadata_dir = os.path.join(self.data_dir, metadata_set)
            metadata_h5_dir = os.path.join(self.data_rootdir, '_h5', metadata_set)
            os.makedirs(metadata_h5_dir, exist_ok=True)

            meta_fns = [fn for fn in os.listdir(metadata_dir) if fn[0] != '.']
            for meta_fn in meta_fns:
                df = pd.read_csv(os.path.join(metadata_dir, meta_fn), header=None)


                with h5py.File(os.path.join(metadata_dir, meta_fn.replace('csv', 'h5')), 'a') as hf:
                    hf.create_dataset(name='mean', data=mean, dtype=np.float32) 




























