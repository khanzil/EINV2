import os
import torch
from torch.utils.data import Dataset, Sampler
import h5py
import numpy as np

def segmentation(num_frames, chunk_len, hop_len):
    start_segment_idx = np.arange(num_frames//hop_len) * hop_len
    stop_segment_idx = np.roll((np.arange(num_frames//hop_len) * hop_len), -1)
    pad_len = []

    if num_frames < (num_frames//hop_len-1)*hop_len + chunk_len:
        pad_len.append((num_frames//hop_len-1)*hop_len + chunk_len - num_frames)
        stop_segment_idx[-1] = (num_frames//hop_len-1)*hop_len + chunk_len
    else:
        stop_segment_idx[-1] = num_frames

    segment_idx = [start_segment_idx, stop_segment_idx]
    return segment_idx, pad_len

class UserDataset(Dataset):
    def __init__(self, cfg, dataset_type=''):
        super.__init__()
        self.data_rootdir = cfg['data rootdir']
        self.dataset = cfg['dataset']['name']
        self.labels = cfg['dataset']['labels']
        self.clip_len = cfg['dataset']['clip_len']
        self.label_res = cfg['dataset']['label_resolution']
        self.sample_rate = cfg['features']['sample_rate']

        metadata_h5_dir = os.path.join(self.data_rootdir, self.dataset, 'metadata')
        features_foldname = "{}_sr{}_nfft{}_hoplen{}_nmels{}".format(cfg['features']['type'], 
                                cfg['features']['sample_rate'], cfg['features']['n_fft'], cfg['features']['hop_len'], cfg['features']['n_mels'])
        if "SALSA" in cfg['features']['type']:
            features_foldname += "_SALSAwinsize{}".format(cfg['features']['SALSA_win_size']) 
        features_h5_dir = os.path.join(self.data_rootdir, self.dataset, '_h5', features_foldname)

        match dataset_type:
            case 'train':
                chunklen = int(cfg['features']['train_chunklen_sec'] * self.sample_rate / cfg['features']['hop_len'])     
                hoplen = int(cfg['features']['train_hoplen_sec'] * self.sample_rate / cfg['features']['hop_len'])
                self.segmented_indexes, self.segmented_pad_width = segmentation(int(self.clip_len*self.sample_rate), chunklen, hoplen)
                fold = str(cfg['train']['train_fold']).split(',')
                overlap = str(cfg['train']['overlap']).split(',')

                meta_rootdir = os.path.join(metadata_h5_dir, cfg['audio_format'] + '_dev')
                fn_rootdir = os.path.join(features_h5_dir, cfg['audio_format'] + '_dev')
                fns = [fn for fn in os.listdir(fn_rootdir) and fn[4] in fold and fn[-1] in overlap]
            
            case 'valid':
                chunklen = int(cfg['features']['train_chunklen_sec'] * self.sample_rate / cfg['features']['hop_len'])     
                hoplen = int(cfg['features']['train_hoplen_sec'] * self.sample_rate / cfg['features']['hop_len'])
                self.segmented_indexes, self.segmented_pad_width = segmentation(int(self.clip_len*self.sample_rate), chunklen, hoplen)
                fold = str(cfg['train']['valid_fold']).split(',')
                overlap = str(cfg['train']['overlap']).split(',')

                meta_rootdir = os.path.join(metadata_h5_dir, cfg['audio_format'] + '_dev')
                fn_rootdir = os.path.join(features_h5_dir, cfg['audio_format'] + '_dev')
                fns = [fn for fn in os.listdir(fn_rootdir) and fn[4] in fold and fn[-1] in overlap]
                
            case 'dev_test':
                chunklen = int(cfg['features']['test_chunklen_sec'] * self.sample_rate / cfg['features']['hop_len'])     
                hoplen = int(cfg['features']['test_hoplen_sec'] * self.sample_rate / cfg['features']['hop_len'])
                self.segmented_indexes, self.segmented_pad_width = segmentation(int(self.clip_len*self.sample_rate), chunklen, hoplen)
                fold = str(cfg['test']['test_fold']).split(',')
                overlap = str(cfg['test']['overlap']).split(',')

                meta_rootdir = os.path.join(metadata_h5_dir, cfg['audio_format'] + '_dev')
                fn_rootdir = os.path.join(features_h5_dir, cfg['audio_format'] + '_dev')
                fns = [fn for fn in os.listdir(fn_rootdir) and fn[4] in fold and fn[-1] in overlap]

            case 'eval_test':
                chunklen = int(cfg['features']['test_chunklen_sec'] * self.sample_rate / cfg['features']['hop_len'])     
                hoplen = int(cfg['features']['test_hoplen_sec'] * self.sample_rate / cfg['features']['hop_len'])
                self.segmented_indexes, self.segmented_pad_width = segmentation(int(self.clip_len*self.sample_rate), chunklen, hoplen)

                fn_rootdir = os.path.join(features_h5_dir, cfg['audio_format'] + '_eval')
                fns = [fn for fn in os.listdir(fn_rootdir)]
            
            case _:
                raise ValueError("This dataset type is not available")
        
        self.dataset_list = []
        for fn in fns:
            for n_segment in range(self.segmented_indexes.shape[1]-1):
                start_idx = self.segmented_indexes[0][n_segment]
                stop_idx = self.segmented_indexes[1][n_segment]
                label_start_idx = start_idx * self.label_res/ (self.sample_rate * cfg['features']['hop_len'])
                label_stop_idx = stop_idx * self.label_res/ (self.sample_rate * cfg['features']['hop_len'])

                with h5py.File(os.path.join(fn_rootdir, fn), 'r') as hf:
                    features = hf['features'][:,start_idx:stop_idx,:]
                pad = ((0,0), (0,self.segmented_pad_width), (0,0))
                features = np.pad(features, pad, mode='constant')

                if 'test' not in dataset_type:
                    label_pad = ((0,self.segmented_pad_width * self.label_res/ (self.sample_rate * cfg['features']['hop_len'])), (0,0), (0,0))
                    with h5py.File(os.path.join(meta_rootdir, fn), 'r') as hf:
                        sed_label = hf['sed_label'][:,label_start_idx:label_stop_idx,:]
                        doa_label = hf['doa_label'][:,label_start_idx:label_stop_idx,:]
                    sed_label = np.pad(sed_label, label_pad, mode='constant')
                    doa_label = np.pad(doa_label, label_pad, mode='constant')
                    
                    self.dataset_list.append({
                        'filename': fn,
                        'n_segment': n_segment,
                        'feature': torch.from_numpy(features),
                        'sed_label': torch.from_numpy(sed_label),
                        'doa_label': torch.from_numpy(doa_label)
                    })
                else:
                    self.dataset_list.append({
                        'filename': fn,
                        'n_segment': n_segment,
                        'feature': torch.from_numpy(features),
                    })

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        return self.dataset_list[idx]
    

class UserBatchSample(Sampler):
    def __init__(self, epoch_size, batch_size, seed):
        super().__init__()        
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(seed=seed)

        self.indexes = np.arange(self.epoch_size)
        self.random_state.shuffle(self.indexes)
        self.pointer = 0

    def set_state(self, sampler_state):
        self.indexes = sampler_state['indexes']
        self.pointer = sampler_state['pointer']
        self.random_state.set_state(sampler_state['random_state'])

    def get_state(self):
        sampler_state = {
            'indexes': self.indexes,
            'pointer': self.pointer,
            'random_state': self.random_state
        }
        return sampler_state
    
    def __iter__(self):
        while True:
            if self.pointer > self.epoch_size:
                self.pointer = 0
                self.random_state.shuffle(self.indexes)
            
            batch_indexes = self.indexes[self.pointer:self.pointer+self.batch_size]
            self.pointer += self.batch_size
            yield batch_indexes

    def __len__(self):
        return (self.epoch_size + self.batch_size - 1) // self.batch_size

class PinMemCustomBatch():
    def __init__(self, batch_dict):
        fn_list = []
        n_segment_list = []
        feature_list = []
        sed_label_list = []
        doa_label_list = []

        for sample in batch_dict:
            fn_list.append(sample['filename'])
            n_segment_list.append(sample['n_segment'])
            feature_list.append(sample['feature'])
            sed_label_list.append(sample['sed_label'])
            doa_label_list.append(sample['doa_label'])
        
        batch_feature = np.stack(feature_list, axis=0)
        batch_sed_label = np.stack(sed_label_list, axis=0)
        batch_doa_label = np.stack(doa_label_list, axis=0)

        self.batch = {
            'filename': fn_list,
            'n_segment': n_segment_list,
            'feature': batch_feature,
            'sed_label': batch_sed_label,
            'doa_label': batch_doa_label
        }

    def pin_memory(self):
        self.batch['feature'] = self.batch['feature'].pin_memory()
        self.batch['sed_label'] = self.batch['sed_label'].pin_memory()
        self.batch['doa_label'] = self.batch['doa_label'].pin_memory()
        return self.batch
    
def collate_fn(batch_dict):
    return PinMemCustomBatch(batch_dict)


class PinMemCustomBatchTest():
    def __init__(self, batch_dict):
        fn_list = []
        n_segment_list = []
        feature_list = []

        for sample in batch_dict:
            fn_list.append(sample['filename'])
            n_segment_list.append(sample['n_segment'])
            feature_list.append(sample['feature'])
        
        batch_feature = np.stack(feature_list, axis=0)

        self.batch = {
            'filename': fn_list,
            'n_segment': n_segment_list,
            'feature': batch_feature,
        }

    def pin_memory(self):
        self.batch['feature'] = self.batch['feature'].pin_memory()
        return self.batch
    
def collate_fn_test(batch_dict):
    return PinMemCustomBatchTest(batch_dict)









