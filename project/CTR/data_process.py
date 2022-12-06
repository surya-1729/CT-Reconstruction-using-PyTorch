import torchvision.datasets as datasets
from Pre_process.sinogram_data_generation import Sinogram_Dataset
from Pre_process.generate_radon import getRadonMatrix

def data_processing(dataset_name, number_of_theta_values, noise_level): 

    # .data as we need only images in the dataset and not the labels
    train_image_data_from_torch_dataset = datasets.dataset_name(root = './data_folder', train = True, download = True).data.double() / 255
    A = getRadonMatrix(number_of_theta_values, (train_image_data_from_torch_dataset[0].shape)) 
    dataset_to_dataloader = Sinogram_Dataset(train_image_data_from_torch_dataset, A, number_of_theta_values, noise_level)

    return dataset_to_dataloader
