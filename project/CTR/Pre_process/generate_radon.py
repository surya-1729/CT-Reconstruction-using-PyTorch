from skimage.transform import radon
import torch
import numpy as np

# Radon Matrix exists for every angle and every line. matrix 
def getRadonMatrix(number_of_theta_values,ny,nx):

    # theta values from 0 and 180 degrees
    theta = np.linspace(0.,180.,number_of_theta_values, endpoint=False)  
    colum_of_A = radon(np.zeros((ny,nx)), theta = theta) # this is only to know the size
    A = np.zeros((colum_of_A.size, nx*ny))
    for i in range(ny):
        for j in range(nx):
            basis_vec = np.zeros((ny,nx))
            basis_vec[i,j] = 1
            colum_of_A = radon(basis_vec, theta = theta)
            A[:,j+i*nx] = np.reshape(colum_of_A, colum_of_A.size)

    return torch.from_numpy(A).double()