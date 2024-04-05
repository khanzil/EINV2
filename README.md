My implementation of EINV2 with modification to run on Kaggle Notebook. Original Source code coude be found at https://github.com/yinkalario/EIN-SELD

# Modification:
Adding
```
   source activate ein
```
in .sh scripts

In seld/methods/ein_seld, change preprocess.py line 148 and training.py line 109 to (this problem could be found in Source code -> Issue)
```
   batch_x = batch_sample.batch_out_dict['waveform']
```
In seld/methods/ein_seld/utils, change SELD_evaluation_metrics_2019.py and SELD_evaluation_metrics_2020.py line 11 to 
```
   eps = np.finfo(float).eps
```  
this is because numpy ver 1.20 remove np.float

In seld/methods/ein_seld/utils, change stft.py line 164 to
```
   fft_window = librosa.util.pad_center(fft_window, size=n_fft)
```
line 86, 87, 134 change ```n``` to ```self.n```
