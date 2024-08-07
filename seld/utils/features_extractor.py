import numpy as np
eps = np.finfo(float).eps
import scipy.signal as ss
import librosa
from ruamel_yaml import YAML
import matplotlib.pyplot as plt
import wave
import os

class afextractor():
    def __init__(self, cfg) -> None:

        features_cfg = cfg['features']
        self.sr = features_cfg['sample rate']
        self.window = features_cfg['window']
        self.n_fft = features_cfg['n_fft']
        self.hop_len = features_cfg['hop_len']
        self.n_mels = features_cfg['n_mels']
        self.n_overlap = self.n_fft - self.hop_len
        self.feature_type = features_cfg['feature type']
        self.SSTF_bin_filter = features_cfg['SS TF bin filter']
        self.SALSA_win_size = features_cfg['SALSA window size']
        self.SALSA_threshold = features_cfg['SALSA threshold']

    def extract_features(self, raw_wave):
        # Calculate channel's stft, raw_wave is numpy
        _,_,stft_W = ss.stft(raw_wave[0, :], fs=self.sr, window=self.window, nperseg=self.n_fft, noverlap=self.n_overlap)
        _,_,stft_Y = ss.stft(raw_wave[1, :], fs=self.sr, window=self.window, nperseg=self.n_fft, noverlap=self.n_overlap)
        _,_,stft_Z = ss.stft(raw_wave[2, :], fs=self.sr, window=self.window, nperseg=self.n_fft, noverlap=self.n_overlap)
        _,_,stft_X = ss.stft(raw_wave[3, :], fs=self.sr, window=self.window, nperseg=self.n_fft, noverlap=self.n_overlap)

        stft = np.stack((stft_W, stft_Y, stft_Z, stft_X), axis=0) # channel x freq x frame        
        spec = np.abs(stft) ** 2

        if self.feature_type == 'LogMel&IVs': # Tested
            # Calculate LogMel Spectrogram
            melW = librosa.filters.mel(sr = 24000, n_fft=self.n_fft, n_mels=self.n_mels, fmin=20, fmax=12000)
            features = np.zeros((spec.shape[0], self.n_mels, spec.shape[2]))
            for channel in range (spec.shape[0]):
                features[channel,:,:] = np.matmul(melW, spec[channel,:,:])
                features[channel,:,:] = librosa.power_to_db(features[channel,:,:], ref=np.max)
            melspec = np.zeros((4,self.n_mels,stft.shape[2]), dtype=complex)
            for chan in range(stft.shape[0]):
                melspec[chan,...] = np.dot(melW, stft[chan,...])
            melspec = np.abs(melspec)**2
            # Calculate Logmel IVs
            IVs = np.zeros((3, features.shape[1], features.shape[2]))

            # y = np.real(np.multiply(spec[0,:,:].conjugate(),spec[1,:,:]))
            # z = np.real(np.multiply(spec[0,:,:].conjugate(),spec[2,:,:]))
            # x = np.real(np.multiply(spec[0,:,:].conjugate(),spec[3,:,:]))

            # for frame in range (features.shape[2]):
            #     norm = np.sqrt(x[:,frame]**2 + y[:,frame]**2 + z[:,frame]**2) + eps
            #     IVs[0,:, frame] = np.dot(melW, y[:,frame]/norm)
            #     IVs[1,:, frame] = np.dot(melW, z[:,frame]/norm)
            #     IVs[2,:, frame] = np.dot(melW, x[:,frame]/norm)
                        
            y = np.real(np.multiply(melspec[0,:,:].conjugate(),melspec[1,:,:]))
            z = np.real(np.multiply(melspec[0,:,:].conjugate(),melspec[2,:,:]))
            x = np.real(np.multiply(melspec[0,:,:].conjugate(),melspec[3,:,:]))

            norm = np.sqrt(x**2 + y**2 + z**2) + eps
            IVs[0,:,:] = y/norm
            IVs[1,:,:] = z/norm
            IVs[2,:,:] = x/norm

            features = np.concatenate((features, IVs), axis=0)

        elif self.feature_type == "LogLin&IVs": # Tested
            features = librosa.power_to_db(spec)

            IVs = np.zeros((3, features.shape[1], features.shape[2]))
            y = np.real(np.multiply(spec[0,:,:].conjugate(),spec[1,:,:]))
            z = np.real(np.multiply(spec[0,:,:].conjugate(),spec[2,:,:]))
            x = np.real(np.multiply(spec[0,:,:].conjugate(),spec[3,:,:]))

            for frame in range (features.shape[2]):
                norm = np.sqrt(x[:,frame]**2 + y[:,frame]**2 + z[:,frame]**2) + eps
            
                IVs[0,:, frame] = y[:,frame]/norm
                IVs[1,:, frame] = z[:,frame]/norm
                IVs[2,:, frame] = x[:,frame]/norm

            features = np.concatenate((features, IVs), axis=0)

        elif self.feature_type == 'SALSA FOA': # Tested
            features = np.zeros(spec.shape)
            for channel in range (spec.shape[0]):
                features[channel,:,:] = librosa.power_to_db(spec[channel,:,:], ref=np.max)
            
            win_size = self.SALSA_win_size
            eigen_inten_vec = np.zeros((stft.shape[0]-1, stft.shape[1], stft.shape[2]))
            stft = np.pad(stft, ((0,0), (0,0), (win_size, win_size)), mode = 'wrap')

            for frame in range (win_size, stft.shape[2]-win_size):
                for freq in range (stft.shape[1]):
                    x = stft[:, freq, frame-win_size:frame+win_size+1]
                    CovMat = np.dot(x, x.conjugate().transpose()) / (1+2*win_size)
                    u,s,_ = np.linalg.svd(CovMat)

                    if self.SSTF_bin_filter is True:
                        if s[0] < s[1] * self.SALSA_threshold:
                            continue

                    normalized_eigen_vec = np.real(u[1:,0]/u[0,0])
                    normalized_eigen_vec = normalized_eigen_vec/(np.linalg.norm(normalized_eigen_vec)+eps)
                    eigen_inten_vec[:,freq,frame-win_size] = normalized_eigen_vec
            
            features = np.concatenate((features, eigen_inten_vec), axis=0)

        elif self.feature_type == 'SALSA MIC': 
            features = np.zeros(spec.shape)
            for channel in range (spec.shape[0]):
                features[channel,:,:] = librosa.power_to_db(spec[channel,:,:], ref=np.max)
            
            stft[:,int(4000*self.n_fft/self.sr):,:] = 0
            win_size = self.SALSA_win_size
            eigen_inten_vec = np.zeros((stft.shape[0]-1, stft.shape[1], stft.shape[2]))

            stft = np.pad(stft, ((0,0), (0,0), (win_size, win_size)), mode = 'wrap')
            for time in range (win_size, stft.shape[2]-win_size):
                for freq in range (stft.shape[1]):
                    x = stft[:, freq, time-win_size:time+win_size+1]
                    CovMat = np.dot(x, x.conjugate().transpose()) / (1+2*win_size)
                    u,s,_ = np.linalg.svd(CovMat)
                    
                    if self.SSTF_bin_filter is True:
                        if s[0] < s[1] * self.SALSA_threshold:
                            continue

                    normalized_eigen_vec = np.angle(u[1:,0] * u[0,0].conjugate())
                    normalized_eigen_vec = normalized_eigen_vec/(2*np.pi*freq*self.sr/(self.n_fft*343)+eps)
                    eigen_inten_vec[:,freq,time-win_size] = normalized_eigen_vec
            
            features = np.concatenate((features, eigen_inten_vec), axis=0)

        elif self.feature_type == "LogMel&GCCPHAT": # Tested
            melW = librosa.filters.mel(sr = 24000, n_fft=self.n_fft, n_mels=self.n_mels, fmin=20, fmax=12000)
            features = np.zeros((spec.shape[0], self.n_mels, spec.shape[2]))
            dis = 0.068
            for channel in range (spec.shape[0]):
                features[channel,:,:] = np.matmul(melW, spec[channel,:,:])
                features[channel,:,:] = librosa.power_to_db(features[channel,:,:], ref=np.max)

            melspec = np.zeros((4,self.n_mels,spec.shape[2]))
            for chan in range(spec.shape[0]):
                melspec[chan,...] = np.dot(melW, spec[chan,...])
            spec = melspec

            gcc_phat = np.zeros((int((stft.shape[0]-1)*stft.shape[0]/2), self.n_mels, stft.shape[2])) 

            for frame in range (stft.shape[2]):
                gcc_chan = 0
                for ichan in range (stft.shape[0]-1):
                    for jchan in range (ichan+1, stft.shape[0]):
                        M = np.ones((int((stft.shape[0]-1)*stft.shape[0]/2), stft.shape[1], 3)) 
                        # if (ichan==0 and jchan==3) or (ichan==1 and jchan==2):    
                        #     dis = 0.0688
                        # else: 
                        #     dis = 0.0685
                        max_delay = int(np.floor(self.sr * dis/343))
                        n = int((self.n_fft*self.n_mels/max_delay)//2)

                        if self.SSTF_bin_filter is False:
                            M = np.ones(stft.shape[1])
                        else:
                            M = self.cdr_cal(fft1=stft[ichan,:,frame], fft2=stft[jchan,:,frame], dis=dis)

                        Rxx = M[gcc_chan,:,frame] * stft[jchan,:,frame] * stft[ichan,:,frame].conjugate()
                        Rt = np.real(np.fft.irfft((Rxx/(np.abs(Rxx)+eps)), n=n))
                        gcc_phat[gcc_chan,:,frame] = np.concatenate((Rt[-self.n_mels//2:], Rt[:self.n_mels//2]))
                        gcc_chan += 1

            features = np.concatenate((features, gcc_phat), axis=0)

        elif self.feature_type == "LogLin&GCCPHAT": # Tested
            features = librosa.power_to_db(spec)
            dis = 0.068
            gcc_phat = np.zeros((int((stft.shape[0]-1)*stft.shape[0]/2), stft.shape[1], stft.shape[2])) 
            for frame in range (stft.shape[2]):
                gcc_chan = 0
                for ichan in range (stft.shape[0]-1):
                    for jchan in range (ichan+1, stft.shape[0]):
                        # if (ichan==0 and jchan==3) or (ichan==1 and jchan==2):    
                        #     dis = 0.0688
                        # else: 
                        #     dis = 0.0685
                        max_delay = int(np.floor(self.sr * dis/343))
                        n = int((self.n_fft**2/max_delay)//4)

                        if self.SSTF_bin_filter is False:
                            M = np.ones(stft.shape[1])
                        else:
                            M = self.cdr_cal(fft1=stft[ichan,:,frame], fft2=stft[jchan,:,frame], dis=dis)

                        Rxx = stft[jchan,:,frame] * stft[ichan,:,frame].conjugate()
                        Rt = np.real(np.fft.irfft((Rxx/(np.abs(Rxx)+eps)), n=n))
                        gcc_phat[gcc_chan,:,frame] = np.concatenate((Rt[-self.n_fft//4:], Rt[:self.n_fft//4+1]))
                        gcc_chan += 1

            features = np.concatenate((features, gcc_phat), axis=0)

        else:
            raise NotImplementedError('This features is not implemented')

        return features
    
    def cdr_cal(self, stft1, stft2, Ffactor = 0.8, cdr_thrs = 0.2, sound_velo = 343, dis=0.068):

        G_mat = np.zeros((3, stft1.shape[1], stft2.shape[2]), dtype=complex)
        M = np.zeros((int((stft1.shape[0]-1)*stft1.shape[0]/2), stft1.shape[1]))

        freqs = np.fft.rfftfreq(self.n_fft, d = 1/self.sr)
        Gnoise = np.sinc(2 * freqs * dis / sound_velo)
        for frame in range (stft1.shape[2]-1):
            G_mat[0,:,frame+1] = (1-Ffactor) * stft1[:,frame] * stft1[:,frame].conjugate() + Ffactor * G_mat[0,:,frame]
            G_mat[1,:,frame+1] = (1-Ffactor) * stft2[:,frame] * stft2[:,frame].conjugate() + Ffactor * G_mat[1,:,frame]
            G_mat[2,:,frame+1] = (1-Ffactor) * stft1[:,frame] * stft2[:,frame].conjugate() + Ffactor * G_mat[2,:,frame]

            Gx = G_mat[2,:,frame+1]/np.sqrt(G_mat[0,:,frame+1]*G_mat[1,:,frame+1]+eps)
            G = np.sqrt(Gnoise**2 * (np.real(Gx)**2 - np.abs(Gx)**2 + 1) - 2 * Gnoise * np.real(Gx) + np.abs(Gx)**2)

            CDR = (Gnoise * np.real(Gx) - np.abs(Gx) ** 2 - G)/(np.abs(Gx)**2 - 1 + eps)
            M[1/(CDR+1) < cdr_thrs] = 1
        
        return M
        
def classwise_logmel():
    yaml = YAML()
    yaml.indent(mapping = 2, sequence=2, offset = 2)
    yaml.default_flow_style = False
    with open('./config.yaml', 'r') as f:
        cfg = yaml.load(f)
    wavname = 'mix001.wav'
    with wave.open(os.path.join('./dataset/foa_eval', wavname), 'rb') as wavfile:
        wav_buf = wavfile.readframes(-1)
        wavdata = np.frombuffer(wav_buf, dtype=np.int16)
        wavdata = wavdata.reshape((int(len(wavdata)/4), 4)).transpose()

    extractor = afextractor(cfg)
    spec = extractor.extract_features(wavdata).numpy()
    # # mix001: 0, 1, 7, 11, 12, 13
    # class_0 = spec[0,:,460*4:500*4]
    # class_1 = spec[0,:,170*4:210*4]
    # class_7 = spec[0,:,390*4:430*4]
    # class_11 = spec[0,:,340*4:380*4]
    # class_12 = spec[0,:,520*4:560*4]
    # class_13 = spec[0,:,10*4:50*4]
    # # mix175: 4, 6, 9
    # class_4 = spec[0,:,20*4:40*4]
    # class_6 = spec[0,:,180*4:220*4]
    # class_9 = spec[0,:,310*4:350*4]
    # # mix189: 2, 3
    # class_2 = spec[0,:,240*4:280*4]
    # class_3 = spec[0,:,280*4:310*4]
    # # mix196: 5, 10
    # class_5 = spec[0,:,330*4:370*4]
    # class_10 = spec[0,:,250*4:290*4]
    # # mix198: 8
    # class_8 = spec[0,:,270*4:310*4]

    # plt.figure(figsize=(10,2))
    # plt.subplot(311)
    # librosa.display.specshow(class_0, sr=24000, y_axis='mel', cmap='jet')
    # plt.title('Alarm')
    # plt.subplot(312)    
    # librosa.display.specshow(class_1, sr=24000, y_axis='mel', cmap='jet')
    # plt.title('Baby cry')
    # plt.subplot(313)
    # librosa.display.specshow(class_11, sr=24000, y_axis='mel', cmap='jet')
    # plt.title('Speech')
    # plt.subplot(414)
    # librosa.display.specshow(class_7, sr=24000, x_axis='time', y_axis='mel', cmap='jet')
    # plt.title('Burning fire')
    # plt.show()

    for chan in range(4):
        plt.subplot(4,1,chan+1)
        librosa.display.specshow(np.asanyarray(spec[chan,:,:]), sr=24000, x_axis='time', cmap='jet')
        plt.yticks([])
    # plt.subplot(211)
    # librosa.display.specshow(np.asanyarray(spec[0,:,:]), sr=24000, y_axis='mel', cmap='jet')
    # plt.subplot(212)
    # melW = librosa.filters.mel(sr = 24000, n_fft=1024, n_mels=256, fmin=20, fmax=12000)
    # librosa.display.specshow(np.asanyarray(spec[4,:,:]), sr=24000, x_axis='time', cmap='jet')
    plt.show()


# classwise_logmel()








