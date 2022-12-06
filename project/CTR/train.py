import torch
import numpy as np
from trainer.trainer import Trainer
from model.models import PrimalDualNetwork, LinearNetwork
from data_process import data_processing
# fix random seeds for reproducibility

SEED = 123
torch.manual_seed(SEED)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main():
    available_datalist = ['MNIST','CIFAR10']
    available_models = [PrimalDualNetwork,LinearNetwork]
    available_optimizers = []
    criterions =[]
    num_train_epochs=1,
    train_batch_size=32, 
    per_device_eval_batch_size=32,   
    noise_level = 0.01
    number_of_theta_values = 26,
    
    data_loader= data_processing(available_datalist[0], number_of_theta_values, noise_level)
    
    trainer = Trainer(
                    available_models[0], 
                    criterions, metrics, available_optimizers,num_train_epochs,
                      data_loader=data_loader,
                     )

    trainer.train()