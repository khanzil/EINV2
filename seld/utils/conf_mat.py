import numpy as np
eps = np.finfo(float).eps
import pandas as pd
import os
import matplotlib.pyplot as plt

classes = {0:'alarm',
           1:'crying baby',
           2:'crash',
           3:'barking dog',
           4:'running engine',
           5:'female scream',
           6:'female speech',
           7:'burning fire',
           8:'footsteps',
           9:'knocking on door',
           10:'male scream',
           11:'male speech',
           12:'ringing phone',
           13:'piano'} 

nb_classes = 14
submissions_dir = './out_infer/ein_seld/EINV2_tPIT_n1/submissions/'
conf_mat = np.zeros((nb_classes+1, nb_classes+1))
# file = pd.read_csv('./dataset/metadata_eval/mix001.csv', header = None)
# print(file.iterrows()[0])

for _, file_name in enumerate(os.listdir(submissions_dir)):
    pred_file = os.path.join(submissions_dir, file_name)
    gt_file = os.path.join('./dataset/metadata_eval',file_name)

    pred_list = pd.read_csv(pred_file, header=None)
    gt_list = pd.read_csv(gt_file, header=None)

    gt_mat = np.zeros((nb_classes, 600))
    pred_mat = np.zeros((nb_classes, 600))

    for row in gt_list.iterrows():
        gt_mat[row[1][1], row[1][0]] = 1
    for row in pred_list.iterrows():
        pred_mat[row[1][1], row[1][0]] = 1

    result = 2*gt_mat - pred_mat

    for idx in range (gt_mat.shape[1]):                
        FP_class = np.argwhere(result[:, idx] == -1) # Pi2
        FN_class = np.argwhere(result[:, idx] == 2) # Ti2
        TP_class = np.argwhere(result[:, idx] == 1) # Ti1/Pi1

        for TP in TP_class:
            conf_mat[TP, TP] += 1

        if (len(FP_class)+len(FN_class)+len(TP_class)) == 0:
            conf_mat[nb_classes, nb_classes] += 1
        elif len(FP_class) == 0:
            for r in FN_class:
                conf_mat[r, nb_classes] += 1
                plt.subplot(FN_class/2, FN_class%2)
                plt.plot()
        elif len(FN_class) == 0:
            if len(TP_class) == 0:
                for c in FP_class:
                    conf_mat[nb_classes, c] += 1
            else:
                for r in TP_class:
                    for c in FP_class:
                        conf_mat[r, c] += 1
        else:
            for r in FN_class:
                for c in FP_class:
                    conf_mat[r, c] += 1

classwise_conf_mat = np.zeros((2, 2, nb_classes))
classwise_f1 = np.zeros((nb_classes,))
for i in range (0, nb_classes):
    classwise_conf_mat[0, 0, i] = conf_mat[i, i]
    classwise_conf_mat[0, 1, i] = np.sum(conf_mat[:, i]) - conf_mat[i,i]
    classwise_conf_mat[1, 0, i] = np.sum(conf_mat[i, :]) - conf_mat[i,i]
    # print(classwise_conf_mat[:,:,i])
    classwise_f1[i] = 2 * classwise_conf_mat[0, 0, i]/(eps + 2 * classwise_conf_mat[0, 0, i] \
                                                    + classwise_conf_mat[0, 1, i] + classwise_conf_mat[1, 0, i])

classwise_f1.reshape((nb_classes, 1))
print(classwise_f1)
classwise_f1 = np.append(classwise_f1, 0)
conf_mat = np.column_stack((conf_mat, classwise_f1))
# df = pd.DataFrame(conf_mat)
# df.to_csv('./result1.csv')

# train_info = np.zeros((nb_classes,))
# for _, file_name in enumerate(os.listdir('./dataset/metadata_dev')):
#     gt_file = os.path.join('./dataset/metadata_dev',file_name)
#     gt_list = pd.read_csv(gt_file, header=None)
#     for row in gt_list.iterrows():
#         train_info[row[1][1]] += 1

# for i in range (len(train_info)):
#     print(int(train_info[i]))


