
from torch.utils.data import Dataset
import torch
import numpy as np

#Custom dataset taking image_dataset ,theta,noise_level and generate a dataset
class Sinogram_Dataset(Dataset):

    def __init__(self,image_data,A, number_of_theta_values,noise_level):  

        self.image_data = image_data
        self.noise_level = noise_level
        self.number_of_theta_values = number_of_theta_values
        self.ny, self.nx = self.image_data[0].shape
        self.A = A  
        
    def __len__(self):
        return len(self.image_data)

    def __getitem__(self,idx):

        label = self.image_data[idx]
        # reshape image matrix into a vector
        img_vector = torch.reshape(self.image_data[idx],(1,self.ny*self.nx))
        sinogram = img_vector@self.A.T
        # generating sinogram image from radonmatrix and reshaping into 2d 
        sinogram = torch.reshape(sinogram, (self.number_of_theta_values,self.ny)) 
        #Adding Noise to the sinogram image 
        sinogram_noisy = sinogram + self.noise_level*torch.randn(sinogram.shape) 

        return sinogram_noisy,label