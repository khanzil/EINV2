import torch
import os
import numpy as np
import random

class CheckPoints():
    def __init__(self, cfg, model, optim, batch_sampler):
        self.checkpoint_list = []
        self.checkpoint_score_list = []
        self.maxlen = cfg['check_point']['maxlen']
        self.remark = cfg['check_point']['remark']
        self.ckpt_dir = "./checkpoints"
        self.model = model
        self.optim = optim
        self.batch_sampler = batch_sampler
          

    def save_ckpt(self, epoch, score, mode):

        if len(self.checkpoint_list) < self.maxlen:
            self.checkpoint_list.append("{}_epoch_{}.pth".format(self.remark, epoch))
            self.checkpoint_score_list.append(score)
        else:
            if mode == 'latest':
                self.checkpoint_list.sort()
                self.delete_ckpt(idx=0)
                self.checkpoint_list.append("{}_epoch_{}.pth".format(self.remark, epoch)) 
                self.checkpoint_score_list.append(score)
            
            elif mode == 'high' and score > np.array(self.checkpoint_score_list).max():
                idx = np.array(self.checkpoint_score_list).argmax()
                self.delete_ckpt(idx)
                self.checkpoint_list.append("{}_epoch_{}.pth".format(self.remark, epoch))
                self.checkpoint_score_list.append(score)
            elif mode == 'low' and score > np.array(self.checkpoint_score_list).min():
                idx = np.array(self.checkpoint_score_list).argmin()
                self.delete_ckpt(idx)
                self.checkpoint_list.append("{}_epoch_{}.pth".format(self.remark, epoch))
                self.checkpoint_score_list.append(score)

    def delete_ckpt(self, idx):
        file_dir = os.path.join(self.ckpt_dir, self.checkpoint_list[idx])
        if os.path.isfile(file_dir):
            os.unlink(file_dir)

        self.checkpoint_list.pop[idx]
        self.checkpoint_score_list.pop[idx]

    def save_file(self, fn, epoch, it):
        file_dir = os.path.join(self.ckpt_dir, fn)
        save_dict = {
            'epoch': epoch,
            'it': it,
            'model': self.model.state_dict(),
            'optim': self.optim.state_dict(),
            'batch_sampler': self.batch_sampler.get_state(),
            'rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state(),
            'random': random.getstate(),
            'np_random': np.random.get_state(),
        }
        torch.save(save_dict, file_dir)

    def load_file(self, fn):
        file_dir = os.path.join(self.ckpt_dir, fn)
        state_dict = torch.load(file_dir)
        epoch = state_dict['epoch']
        it = state_dict['it']
        self.model.load_state_dict(state_dict['model'])
        self.optim.load_state_dict(state_dict['optim'])
        self.batch_sampler.set_state(state_dict['batch_sampler'])
        torch.set_rng_state(state_dict['rng'])
        torch.cuda.set_rng_state(state_dict['cuda_rng'])
        random.setstate(state_dict['random'])
        np.random.set_state(state_dict['np_random'])
        return epoch, it





ckpt = CheckPoints(5, 'latest')

for i in range (7):
    ckpt.append("File {}".format(i), i)
print(ckpt.checkpoint_list)



