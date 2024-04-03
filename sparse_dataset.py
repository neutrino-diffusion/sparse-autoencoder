from pyexpat import features
import h5py
import matplotlib.pyplot as plt
import numpy as np
import MinkowskiEngine as ME
import torch
import seaborn as sns
sns.set_context('talk')


class NearDetDataset3D(torch.utils.data.Dataset):
    def __init__(
        self, 
        input_size=512,
        scaled_down = 32, 
        root='/global/homes/r/rradev/sparse_autoencoder/data/pointcloud_output.hdf5'):
        super().__init__()
        self.input_size = input_size
        self.pixel_pitch =  0.4434
        self.cm2mm = 10
        self.data = h5py.File(root, mode='r+')
        self.valid_indices = list(self.data.keys())
        self.scaled_down = scaled_down

    def __getitem__(self, index):
        event_key = self.valid_indices[index]
        
        x, y, z, adc = self.data[f'{event_key}'][:].T

        adc = adc/255.0
        quantized_coordinates, quantized_features = self.get_coordinates_features(x, y, z, adc)

        return {
            'input': (quantized_coordinates, quantized_features),
            'target': (quantized_coordinates, quantized_features),
        }
    
    def __len__(self):
        return len(self.valid_indices)

    def quantize_coordinates(self, coordinates, features, quantization_factor):
        quantization_size = (self.pixel_pitch * self.cm2mm) * quantization_factor
        quantized_coordinates, quantized_features = ME.utils.sparse_quantize(
                coordinates=coordinates,
                features=features,
                quantization_size=quantization_size
        )

        return quantized_coordinates, quantized_features


    def shift_coordinates(self, x, y, z):
        x_min = np.min(x)
        y_min = np.min(y)
        z_min = np.min(z)
        return (x - x_min, y - y_min, z - z_min)


    def trim_coordinates(self, x, y, z, adc):
        size =  self.pixel_pitch * self.cm2mm * self.input_size
        coordinates = np.array([x, y, z, adc])
        cond = np.all(coordinates < size, axis=0)
        return coordinates[:, cond]


    def get_coordinates_features(self, x, y, z, adc):
        x, y, z = self.shift_coordinates(x, y, z)
        x, y, z, adc = self.trim_coordinates(x, y, z, adc)
        coordinates = torch.tensor(
            np.array([x, y, z]).T
        ).contiguous()
        features = np.expand_dims(adc, axis=1)
        scale_factor = self.scaled_down
        quantized_coordinates, quantized_features = self.quantize_coordinates(coordinates, features, scale_factor)
        return quantized_coordinates, quantized_features


def create_voxel_array(coords, max_voxel_size):
    voxelarray = np.zeros(max_voxel_size, dtype=bool)
    for coord in coords:
        voxelarray[coord[0], coord[1], coord[2]] = True
    return voxelarray


if __name__ == "__main__":
    dataset = NearDetDataset3D()
    print(len(dataset))

    # Maximum voxel size
    max_voxel_size = (16, 16, 16)
    for i in range(20):
        coords = dataset[i]['input'][0]
        voxel_array = create_voxel_array(coords, max_voxel_size)
        # plot in 3D 
        # Create the voxel array

        # Plot
        plt.rcParams["figure.figsize"] = (10, 10)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.voxels(voxel_array)
        # add shadows
        ax.view_init(30, 30)
        plt.savefig(f'voxel_{i}.png')
