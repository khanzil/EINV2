from ruamel_yaml import YAML
from seld.learning.preprocess import Preprocessor
import h5py
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd
from seld.utils.features_extractor import afextractor
yaml = YAML()
yaml.indent(mapping = 2, sequence=2, offset = 2)
yaml.default_flow_style = False
with open('./configs/config.yaml', 'r') as f:
    cfg = yaml.load(f)

with h5py.File('./dataset_root/TNSSE2020/_h5/metadata_dev/fold1_room1_mix001_ov1.h5', 'r') as hf:
    print(hf['sed_label'].shape)
    features = hf['sed_label'][()]

plt.plot(features[:,0,5])
plt.show()
# for chan in range(features.shape[0]):
#     plt.subplot(features.shape[0], 1, chan+1)
#     librosa.display.specshow(features[chan], sr=24000, y_axis='mel', cmap='jet')

# df = pd.read_csv('./dataset_root/TNSSE2020/metadata/metadata_dev/fold1_room1_mix001_ov1.csv', header=None)
# for row in df.iterrows():
#     print(row[0])







