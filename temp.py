from ruamel_yaml import YAML
from seld.learning.preprocess import Preprocessor
import h5py
import numpy as np
import librosa
import matplotlib.pyplot as plt
from seld.utils.features_extractor import afextractor
yaml = YAML()
yaml.indent(mapping = 2, sequence=2, offset = 2)
yaml.default_flow_style = False
with open('./configs/config.yaml', 'r') as f:
    cfg = yaml.load(f)

# preproc = Preprocessor(cfg)
# preproc.extract_features(cfg)

# with h5py.File('./dataset_root/TNSSE2020/LogMel_IVs_sr24000_nfft1024_hoplen600_nmels256/features/foa_dev/fold1_room1_mix001_ov1.h5', 'r') as hf:
#     print(hf['feature'].shape)
#     features = hf['feature'][()]


# for chan in range(features.shape[0]):
#     plt.subplot(features.shape[0], 1, chan+1)
#     librosa.display.specshow(features[chan], sr=24000, y_axis='mel', cmap='jet')

# plt.show()

x = np.arange(10)
print(np.where(x<4))