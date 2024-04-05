My implementation of EINV2 with modification to run on Kaggle Notebook. Original Source code could be found at https://github.com/yinkalario/EIN-SELD

# Modification:
Adding
```
   source activate ein
```
in .sh scripts

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
