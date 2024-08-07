from seld.utils.features_extractor import afextractor
from ruamel.yaml import YAML
import shutil
import os
from tqdm import tqdm
import librosa
import numpy as np
import h5py

from timeit import default_timer as timer

yaml = YAML()
with open('./config.yaml', 'r') as f:
    cfg = yaml.load(f)


def extract_features(config =  cfg):

        yaml = YAML()
        yaml.indent(mapping = 2, offset = 2)
        yaml.default_flow_style = False
        with open('./config.yaml', 'r') as f:
            config = yaml.load(f)

        audio_format = config['features']['audio_format']
        sr = config['features']['sample rate']
        stft_feature_extractor = afextractor(config)
        if audio_format == 'foa':
            types = ['foa_dev']
        elif audio_format == 'eval':
            types = ['mic_dev', 'mic_eval']
        
        start_time = timer()
        for type in types:
            audio_dir = os.path.join(config['data_dir'], type )
            feature_dir = os.path.join(config['feature_dir'], config['features']['type'], audio_format)

            shutil.rmtree(feature_dir, ignore_errors = True)
            os.makedirs(feature_dir, exist_ok = True)
            
            audio_list = sorted(os.listdir(audio_dir))
            for i, n in enumerate(tqdm(audio_list)):
                full_audio_fn = os.path.join(audio_dir, n)
                audio_input,_ = librosa.load(full_audio_fn,sr=sr, mono=False, dtype = np.float32)

                everything_feature = stft_feature_extractor.extract_features(audio_input)
                feature_final = os.path.join(feature_dir, n.replace('wav', 'h5'))
                with h5py.File(feature_final, 'w') as hf:
                    hf.create_dataset('feature', data = everything_feature, dtype =np.float32)
                tqdm.write('{},{},{}'.format(i, n, everything_feature.shape))
        tqdm.write('Done with extraction')
        print("After {:.3f}".format(timer() - start_time))


print(cfg['feature_dir'])
extract_features()
