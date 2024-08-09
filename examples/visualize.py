import numpy as np
import wave
import librosa
from seld.utils.data.DataAug import *
import torch
from ruamel.yaml import YAML
import matplotlib.pyplot as plt
import soundfile as sf
import time
import h5py
import os
import torchaudio
import scipy.signal as ss
import torchaudio.transforms as tf
import pandas as pd
# from utils.features import afextractor
eps = np.finfo(float).eps

# metalist = os.listdir('./dataset/metadata_dev')
# ov2 = [path for path in metalist if 'ov2' in path]
# print(len(ov2))

# yaml = YAML()
# yaml.indent(mapping = 2, sequence=2, offset = 2)
# yaml.default_flow_style = False
# with open('./config.yaml', 'r') as f:
#     cfg = yaml.load(f)

# with h5py.File('fold1_room1_mix001_ov1.h5', 'r') as file:
#     print(file.keys())
#     print(file['feature'][()].shape)
#     x = file['feature'][()]

# for chan in range (x.shape[0]):
#     plt.subplot(x.shape[0],1,chan+1)
#     librosa.display.specshow(np.asanyarray(x[chan,:,:]).T, sr=24000, x_axis='time', y_axis='linear', cmap='jet')

# plt.show()

# wavdata, sr = torchaudio.load('dataset/foa_dev/fold1_room1_mix001_ov1.wav')
# window = 'hann'
# f,t,stft = ss.stft(wavdata[0, :sr*5], fs=sr, window=window, nperseg=512, noverlap=212)
# spec = np.abs(stft)**2
# mel = librosa.filters.mel(sr = 24000, n_fft=512, n_mels=128, fmin=20, fmax=12000)
# spec2 = librosa.power_to_db(np.matmul(mel,spec))
# spec1 = librosa.power_to_db(spec)
# win1 = np.zeros(sr*5)
# win1[300*100:300*120] = 1
# win2 = np.zeros(10)

# transform = transforms.SpecAugment(n_freq_masks=1, freq_mask_param=20,\
#                                            n_time_masks=1, time_mask_param=15, p=0.5)
# spec_tensor = torch.from_numpy(spec)
# specaug = transform(spec_tensor)
# specaug = librosa.power_to_db(specaug)
# plt.subplot(311)
# plt.grid()
# plt.xlim(0,sr*5)
# plt.plot(wavdata[0,:sr*5])
# plt.title('Waveform')
# plt.xticks([])
# plt.twinx()
# plt.plot(win1, color='red')
# plt.xticks([])
# plt.yticks([])
# plt.subplot(312)
# librosa.display.specshow(np.asanyarray(spec1), sr=24000, y_axis='linear', cmap='jet')
# plt.title('Spectrogram')
# plt.twinx()
# plt.xlim(0,len(win2))
# plt.ylim(0,1)
# plt.yticks([])
# plt.plot(win2)


plt.hlines(y=1, xmin=10, xmax=120, colors='r')
plt.hlines(y=0, xmin=40, xmax=90, colors='g')
plt.hlines(y=2, xmin=140, xmax=170, colors='b')
plt.legend(['Cảnh báo', 'Em bé khóc', 'Tiếng nói'], loc='lower right')
plt.xlabel('Thời gian')
plt.ylabel('Góc tới')
plt.yticks([])
plt.xlim([0,200])
plt.grid()
plt.show()


# plt.subplot(313)
# librosa.display.specshow(np.asanyarray(spec2), sr=24000, x_axis='time', y_axis='mel', cmap='jet')
# plt.title('Mel Spectrogram')
# plt.show()

# with h5py.File('comparison_dev1.h5', 'r') as file:
#     spec = file['spec'][()]
#     SALSA = file['SALSA'][()]
#     SALSA_mask = file['SALSA mask'][()]
#     DA = file['DA'][()]
#     DA_mask = file['DA mask'][()]

# csv = pd.read_csv('./dataset/metadata_dev/fold1_room1_mix006_ov1.csv')
# # csv = pd.read_csv('./out_infer/Proposed_mix002.csv')

# # gt = np.zeros((2401,2))
# # active_frame = np.zeros(2401)
# gt = np.zeros((600,2))
# active_frame = np.zeros((600,14))

# for row in csv.iterrows():
#     # active_frame[row[1].iloc[0]*4:(row[1].iloc[0]+1)*4] = 1  
#     # gt[row[1].iloc[0]*4:(row[1].iloc[0]+1)*4,0] = row[1].iloc[3] * np.pi/180 # azi
#     # gt[row[1].iloc[0]*4:(row[1].iloc[0]+1)*4,1] = row[1].iloc[4] * np.pi/180 # ele

#     active_frame[row[1].iloc[0], row[1].iloc[1]] = row[1].iloc[1]

#     gt[row[1].iloc[0],0] = row[1].iloc[2] * np.pi/180 # azi
#     gt[row[1].iloc[0],1] = row[1].iloc[3] * np.pi/180 # ele

# for frame in range(gt.shape[0]-1):
#     active_frame[frame,:] = active_frame[frame+1,:] - active_frame[frame,:]

# for label in range (active_frame.shape[1]):
#     if np.where(active_frame[:,label]==label)[0].size==0:
#         continue
#     start = np.where(active_frame[:,label]==label)[0]
#     stop = np.where(active_frame[:,label]==-label)[0]

#     match label:
#         case 0:
#             color = 'b'
#         case 1:
#             color = 'g'
#         case 2:
#             color = 'r'
#         case 3:
#             color = 'c'
#         case 4:
#             color = 'm'
#         case 5:
#             color = 'pink'
#         case 6:
#             color = 'k'
#         case 7:
#             color = 'k'
#         case 8:
#             color = 'orange'
#         case 9:
#             color = 'brown'
#         case 10:
#             color = 'purple'
#         case 11:
#             color = 'olive'
#         case 12:
#             color = 'b'
#         case 13:
#             color = 'navy'    

#     for cnt, start_idx in enumerate(start):
#         plt.hlines(y=label, xmin=start_idx, xmax=stop[cnt], colors=color)
# plt.grid()
# plt.xlabel('Time')
# plt.ylabel('Class')
# plt.show()

# ele = np.arcsin(SALSA[1,:,:])
# azix = np.arccos(np.clip(SALSA[2]/np.cos(ele),-1,1))
# aziy = np.arcsin(np.clip(SALSA[0]/np.cos(ele),-1,1))
# azi = np.zeros(SALSA.shape[1:])
# SALSA_mask[:,active_frame==0] = 0
# SALSA_mask[spec[0,:,:]<-90] = 0

# le = 0

# for frame in range(SALSA.shape[2]):
#     for freq in range(SALSA.shape[1]):
#         # if SALSA_mask[freq,frame] == 0:
#         #     continue

#         if (aziy[freq,frame] < 0):
#             azi[freq,frame] = - azix[freq,frame]
#         else:
#             azi[freq,frame] = azix[freq,frame]

#         dist1 = np.sin(ele[freq,frame]) * np.sin(gt[frame,1]) + np.cos(ele[freq,frame]) * np.cos(gt[frame,1]) * np.cos(azi[freq,frame] - gt[frame,0])

#         dist1 = np.arccos(np.clip(dist1, -1, 1))*180/np.pi
#         le += dist1

# le = le/np.sum(SALSA_mask)   
# print(le)

# SALSA_le = (44.53+61.87+30.97+24.43+31.31+30.88+30.04+28.51+40.75)/9
# SALSA_le = (33.58+38.58+22.78+21.81+20.79+23.66+24.27+20.67+25.79)/9
# print(SALSA_le)

# DA_le = (45.81+67.90+32.63+25.55+33.81+32.74+30.78+30.61+45.01)/9
# DA_le = (33.90+43.90+23.89+22.05+21.58+24.70+24.98+21.50+26.58)/9
# print(DA_le)

# plt.plot(azi[100,:])
# plt.plot(gt[:,0])
# plt.show()


# plt.subplot(421)
# librosa.display.specshow(spec[0,:,40*30:40*40], sr=24000, y_axis='linear', cmap='jet')
# plt.title('SALSA')

# plt.subplot(422)
# librosa.display.specshow(spec[0,:,40*30:40*40], sr=24000, y_axis='linear', cmap='jet')
# plt.title('DA')

# plt.subplot(423)
# librosa.display.specshow((SALSA[0,:,40*30:40*40]), sr=24000, y_axis='linear', cmap='jet')

# plt.subplot(425)
# librosa.display.specshow((SALSA[1,:,40*30:40*40]), sr=24000, y_axis='linear', cmap='jet')

# plt.subplot(427)
# librosa.display.specshow((SALSA[2,:,40*30:40*40]), sr=24000, x_axis='time', y_axis='linear', cmap='jet')

# plt.subplot(424)
# librosa.display.specshow((DA[0,:,40*30:40*40]), sr=24000, y_axis='linear', cmap='jet')

# plt.subplot(426)
# librosa.display.specshow((DA[1,:,40*30:40*40]), sr=24000, y_axis='linear', cmap='jet')

# plt.subplot(428)
# librosa.display.specshow((DA[2,:,40*30:40*40]), sr=24000, x_axis='time', y_axis='linear', cmap='jet')

# plt.subplot(311)
# librosa.display.specshow(spec[0,:,40*32:40*37], sr=24000, y_axis='linear', cmap='jet')
# plt.title('Spectrogram')

# plt.subplot(312)
# librosa.display.specshow(SALSA[0,:,40*32:40*37], sr=24000, y_axis='linear', cmap='binary_r')
# plt.title('SALSA')

# plt.subplot(313)
# librosa.display.specshow(DA[0,:,40*32:40*37], sr=24000, x_axis = 'time', y_axis='linear', cmap='binary_r')
# plt.title('Proposed')

# plt.show()








