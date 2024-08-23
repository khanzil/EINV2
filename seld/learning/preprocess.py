import os
import tqdm
import h5py
import librosa
import numpy as np
import pandas as pd
from seld.utils.data.features_extractor import afextractor

class Preprocessor():
    def __init__(self, cfg):
        self.data_rootdir = cfg['data_rootdir']
        self.dataset = cfg['dataset']['name']

        self.data_dir = os.path.join(self.data_rootdir, self.dataset, 'data')

    def extract_features(self, cfg):
        features_foldname = "{}_sr{}_nfft{}_hoplen{}_nmels{}".format(cfg['features']['type'], 
                                cfg['features']['sample_rate'], cfg['features']['n_fft'], cfg['features']['hop_len'], cfg['features']['n_mels'])
        if "SALSA" in cfg['features']['type']:
            features_foldname += "_SALSAwinsize{}".format(cfg['features']['SALSA_win_size']) 

        features_h5_dir = os.path.join(self.data_rootdir, self.dataset, '_h5', features_foldname)
        assert os.path.isdir(features_h5_dir) == False, f'{features_h5_dir} is already exists'

        total = []

        if cfg['audio_format'] == 'foa':
            audio_sets = ['foa_dev', 'foa_eval']
        if cfg['audio_format'] == 'mic':
            audio_sets = ['mic_dev', 'mic_eval']
        extractor = afextractor(cfg)

        for audio_set in audio_sets:
            os.makedirs(os.path.join(features_h5_dir, audio_set))
            wav_fns = [fn for fn in os.listdir(os.path.join(self.data_dir, audio_set)) if ".wav" in fn]

            for wav_fn in wav_fns:
                waveform, _ = librosa.load(os.path.join(self.data_dir, audio_set, wav_fn), sr=cfg['features']['sample_rate'], mono=False, dtype=np.float32)
                features = extractor.extract_features(waveform)
                total.append(features)
                h5_fn = wav_fn.replace('wav', 'h5')
                with h5py.File(os.path.join(features_h5_dir, audio_set, h5_fn), 'w') as hf:
                    hf.create_dataset(name='features', data=features, dtype=np.float32)

        total = np.stack(total, axis=0)
        mean = np.squeeze(np.mean(total, axis=(0,4), keepdims=True), axis=0)
        std = np.squeeze(np.std(total, axis=(0,4), keepdims=True), axis=0)

        with h5py.File(os.path.join(features_h5_dir, 'scalar.h5'), 'w') as hf:
            hf.create_dataset(name='mean', data=mean, dtype=np.float32)        
            hf.create_dataset(name='std', data=std, dtype=np.float32)        

    def extract_metadata(self):
        if self.dataset != 'TNSSE2020':
            return

        num_frames = 600
        num_classes = 14
        num_tracks = 2
        deg2rad = np.pi/180

        metadata_sets = ['metadata_dev', 'metadata_eval']
        for metadata_set in metadata_sets:
            metadata_dir = os.path.join(self.data_dir, metadata_set)
            metadata_h5_dir = os.path.join(self.data_rootdir, self.dataset, '_h5', metadata_set)
            os.makedirs(metadata_h5_dir, exist_ok=True)

            meta_fns = [fn for fn in os.listdir(metadata_dir) if fn[0] != '.']
            for meta_fn in meta_fns:
                df = pd.read_csv(os.path.join(metadata_dir, meta_fn), header=None)
                sed_label = np.zeros((num_frames, num_tracks, num_classes))
                doa_label = np.zeros((num_frames, num_tracks, 3))
                for row in df.iterrows():
                    frame = row[1][0]
                    event = row[1][1]
                    track = row[1][2]
                    azi = row[1][3]
                    ele = row[1][4]
                    
                    sed_label[frame, track, event] = 1
                    doa_label[frame, track, :] = np.cos(azi*deg2rad) * np.cos(ele*deg2rad),\
                                        np.sin(azi*deg2rad) * np.cos(ele*deg2rad), np.sin(ele)

                with h5py.File(os.path.join(metadata_h5_dir, meta_fn.replace('csv', 'h5')), 'a') as hf:
                    hf.create_dataset(name='sed_label', data=sed_label, dtype=np.float32) 
                    hf.create_dataset(name='doa_label', data=doa_label, dtype=np.float32) 
            

            


























