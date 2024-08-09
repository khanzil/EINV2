import numpy as np
import random
import torch 
import torch.nn as nn
import scipy.signal as ss
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader 
from seld.utils.data.features_extractor import *
from ruamel.yaml import YAML
import time
import soundfile as sf
import rir_generator
import time
eps = np.finfo(float).eps
import h5py
# rad = 0.042
deg35 = 35*np.pi/180
deg45 = 45*np.pi/180

# M1 M4: 0.0688
# M1 M2/M3: 0.0685

window = 'hann'
n_fft = 1024
n_overlap = 424
win_size = 3

yaml = YAML()
yaml.indent(mapping = 2, sequence=2, offset = 2)
yaml.default_flow_style = False
with open('./config.yaml', 'r') as f:
    cfg = yaml.load(f)



# SIMULATIONs


# def coh_test(stft, threshold = 5):
#     drr = np.zeros(stft.shape[1:])
#     mask = np.zeros(stft.shape[1:])
#     mask2 = np.zeros(stft.shape[1:])
#     win_size2 = 1
#     stft3 = np.pad(stft, ((0,0),(win_size2,win_size2),(win_size2,win_size2)), mode='wrap')
#     stft2 = np.pad(stft, ((0,0),(win_size,win_size),(0,0)), mode='wrap')
#     stft1 = np.pad(stft, ((0,0),(0,0),(win_size,win_size)), mode='wrap')

#     fea_mat1 = np.zeros((3,stft.shape[1],stft.shape[2]))
#     fea_mat2 = np.zeros((3,stft.shape[1],stft.shape[2]))

#     start = time.time()
#     for frame in range (win_size, stft1.shape[2]-win_size):
#         for freq in range (stft1.shape[1]):
#             x = stft1[:, freq, frame-win_size:frame+win_size+1]
#             CovMat = np.dot(x, x.conjugate().transpose()) / float(1+2*win_size)
#             u,s,_ = np.linalg.svd(CovMat)
#             if s[0] > s[1]*threshold:
#                 mask[freq,frame-win_size] = 1
#             normed_eigenvector = np.real(u[1:, 0] / u[0, 0])
#             normed_eigenvector = normed_eigenvector/np.sqrt(np.sum(normed_eigenvector**2))
#             fea_mat1[:,freq,frame-win_size] = normed_eigenvector
#     print(time.time()-start)
#     start = time.time()
#     for frame in range (win_size2, stft3.shape[2]-win_size2):
#         for freq in range (win_size2, stft3.shape[1]-win_size2):
#             x = stft3[:, freq-win_size2:freq+win_size2+1, frame-win_size2:frame+win_size2+1].reshape((4,-1))
#             CovMat = np.dot(x, x.conjugate().transpose()) / (1+2*win_size2)**2
#             u,s,_ = np.linalg.svd(CovMat)
#             if s[0] > s[1]*threshold:
#                 mask2[freq-win_size2,frame-win_size2] = 1
#             normed_eigenvector = np.real(u[1:, 0] / u[0, 0])
#             normed_eigenvector = normed_eigenvector/np.sqrt(np.sum(normed_eigenvector**2))
#             fea_mat2[:,freq-win_size2,frame-win_size2] = normed_eigenvector
#     print(time.time()-start)
#     return mask, fea_mat1, mask2, fea_mat2

# def add_rir(x, sr, azi, ele, rev_time = 0.3, L=[3,4,5], r=[1.5,2,1]):
#     rir_w = rir_generator.generate(
#         c=340,
#         fs=sr,
#         L=L,
#         reverberation_time=rev_time,
#         nsample=int(0.3*sr),
#         r=r,
#         s=np.add(r,[np.cos(np.pi*azi/180),np.sin(np.pi*azi/180),np.sin(np.pi*ele/180)]),
#     )

#     rir_y = rir_generator.generate(
#         c=340,
#         fs=sr,
#         L=L,
#         reverberation_time=rev_time,
#         nsample=int(0.3*sr),
#         r=r,
#         s=np.add(r,[np.cos(np.pi*azi/180),np.sin(np.pi*azi/180),np.sin(np.pi*ele/180)]),
#         mtype=rir_generator.mtype.bidirectional,
#         orientation=[np.pi/2, 0]
#     )

#     rir_z = rir_generator.generate(
#         c=340,
#         fs=sr,
#         L=L,
#         reverberation_time=rev_time,
#         nsample=int(0.3*sr),
#         r=r,
#         s=np.add(r,[np.cos(np.pi*azi/180),np.sin(np.pi*azi/180),np.sin(np.pi*ele/180)]),
#         mtype=rir_generator.mtype.bidirectional,
#         orientation=[0, np.pi/2]
#     )

#     rir_x = rir_generator.generate(
#         c=340,
#         fs=sr,
#         L=L,
#         reverberation_time=rev_time,
#         nsample=int(0.3*sr),
#         r=r,
#         s=np.add(r,[np.cos(np.pi*azi/180),np.sin(np.pi*azi/180),np.sin(np.pi*ele/180)]),
#         mtype=rir_generator.mtype.bidirectional,
#     )
#     x = np.pad(x, (len(rir_w),0))
#     w = np.convolve(x,rir_w[:,0],mode='valid')
#     y = np.convolve(x,rir_y[:,0],mode='valid')
#     z = np.convolve(x,rir_z[:,0],mode='valid')
#     x = np.convolve(x,rir_x[:,0],mode='valid')
#     return np.concatenate([w,y,z,x]).reshape(4,len(w))

# if 0:
#     wavdata, sr = sf.read('dataset/LibriTTS/dev-clean/251/118436/251_118436_000021_000003.wav', dtype=np.int16)
#     wavdata1 = wavdata[sr*0:sr*2]

#     wavdata, sr = sf.read('dataset/LibriTTS/dev-clean/422/122949/422_122949_000013_000010.wav', dtype=np.int16)
#     wavdata2 = wavdata[sr*0:sr*2]

#     noise, noise_sr = sf.read('dataset/NOISEX92/white/signal.wav', dtype=np.int16)
#     noise = ss.resample(noise, int(len(noise)/noise_sr*sr))

#     wavdata1 = add_rir(wavdata1, sr, azi=110, ele=-10)
#     wavdata2 = add_rir(wavdata2, sr, azi=170, ele=15)
#     wavdata = wavdata1 + wavdata2

#     _,_,spec1 = ss.stft(wavdata1[0,:], fs=sr, window=window, nperseg=n_fft, nfft=n_fft, noverlap=n_overlap)
#     _,_,spec2 = ss.stft(wavdata2[0,:], fs=sr, window=window, nperseg=n_fft, nfft=n_fft, noverlap=n_overlap)

#     spec1 = np.abs(spec1) ** 2
#     spec1 = librosa.power_to_db(spec1)
#     spec2 = np.abs(spec2) ** 2
#     spec2 = librosa.power_to_db(spec2)

#     for chan in range(wavdata.shape[0]):
#         noise_idx = np.random.choice((len(noise)-wavdata.shape[1]))
#         wavdata[chan,:] += noise[noise_idx:noise_idx+wavdata.shape[1]]/1000

# else:
#     wavdata, sr = sf.read('dataset/foa_eval/mix001.wav', dtype=np.float32)
#     wavdata = wavdata.transpose()


# EXTRACTOR TEST


# _,_,stft_W = ss.stft(wavdata[0,:], fs=sr, window=window, nperseg=n_fft, nfft=n_fft, noverlap=n_overlap)
# _,_,stft_Y = ss.stft(wavdata[1,:], fs=sr, window=window, nperseg=n_fft, nfft=n_fft, noverlap=n_overlap)
# _,_,stft_Z = ss.stft(wavdata[2,:], fs=sr, window=window, nperseg=n_fft, nfft=n_fft, noverlap=n_overlap)
# _,_,stft_X = ss.stft(wavdata[3,:], fs=sr, window=window, nperseg=n_fft, nfft=n_fft, noverlap=n_overlap)

# stft = np.stack((stft_W, stft_Y, stft_Z, stft_X), axis=0) # channel x freq x time
# spec = np.abs(stft) ** 2
# spec = librosa.power_to_db(spec)

# mask_drr1, fea_mat1, mask_drr2, fea_mat2 = coh_test(stft=stft)

# drr_error = np.sum(mask_drr*le)/np.sum(mask_drr)
# drr2_error = np.sum(mask_drr2*le)/np.sum(mask_drr2)
# drr3_error = np.sum(mask_drr3*le)/np.sum(mask_drr3)
# lrss_error = np.sum(mask_lrss*le)/np.sum(mask_lrss)
# print("{}, {}, {}, {}".format(drr_error, drr2_error, drr3_error, lrss_error))

# plt.subplot(611)
# librosa.display.specshow(np.asanyarray(spec1), sr=sr, x_axis='time', y_axis='linear', cmap='jet')
# plt.subplot(612)
# # librosa.display.specshow(np.asanyarray(spec2), sr=sr, x_axis='time', y_axis='linear', cmap='jet')
# plt.subplot(311)
# librosa.display.specshow(np.asanyarray(spec[0,:,:]), sr=sr, x_axis='time', y_axis='linear', cmap='jet')
# plt.subplot(312)
# librosa.display.specshow(np.asanyarray(mask_drr1), sr=sr, x_axis='time', y_axis='linear', cmap='binary')
# plt.subplot(313)
# librosa.display.specshow(np.asanyarray(mask_drr2), sr=sr, x_axis='time', y_axis='linear', cmap='binary')

# plt.subplot(411)
# librosa.display.specshow(np.asanyarray(spec[0,:,:]), sr=sr, x_axis='time', y_axis='linear', cmap='jet')
# plt.subplot(412)
# librosa.display.specshow(np.asanyarray(fea_mat1[0,:,:]), sr=sr, x_axis='time', y_axis='linear', cmap='jet')
# plt.subplot(413)
# librosa.display.specshow(np.asanyarray(fea_mat1[0,:,:]), sr=sr, x_axis='time', y_axis='linear', cmap='jet')
# plt.subplot(414)
# librosa.display.specshow(np.asanyarray(fea_mat1[0,:,:]), sr=sr, x_axis='time', y_axis='linear', cmap='jet')

# with h5py.File('./comparison2.h5', 'w') as hf:
#     hf.create_dataset('spec', data=spec, dtype=float)
#     hf.create_dataset('SALSA', data=fea_mat1, dtype=float)
#     hf.create_dataset('DA', data=fea_mat2, dtype=float)
#     hf.create_dataset('SALSA mask', data=mask_drr1, dtype=float)
#     hf.create_dataset('DA mask', data=mask_drr2, dtype=float)