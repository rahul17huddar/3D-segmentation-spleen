from monai.utils import first
from monai.transforms import(
    Compose,
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    Activations,
)

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.data import DataLoader, Dataset
import torch
from PIL import Image

import numpy as np

from monai.inferers import sliding_window_inference

class MONAI:
    def __init__(self, weights, device, test_files, slice_no):
        self.weights = weights
        self.device = device
        self.slice_no = slice_no - 1
        self.test_files = test_files

    def ViewImage(self):
        test_transforms = Compose(
                [
                    LoadImaged(keys=["vol"]),
                    AddChanneld(keys=["vol"]),
                    Spacingd(keys=["vol"], pixdim=(1.5,1.5,1.0), mode=("bilinear")), 
                    Orientationd(keys=["vol"], axcodes="RAS"),
                    ScaleIntensityRanged(keys=["vol"], a_min=-200, a_max=200,b_min=0.0, b_max=1.0, clip=True), 
                    CropForegroundd(keys=['vol'], source_key='vol'),
                    Resized(keys=["vol"], spatial_size=[128,128,64]),   
                    ToTensord(keys=["vol"]),
                ]
            )
    
        test_ds = Dataset(data=self.test_files, transform=test_transforms)
        self.test_loader = DataLoader(test_ds, batch_size=1)
        test_patient = first(self.test_loader)

        return np.asarray(test_patient["vol"][0, 0, :, :, self.slice_no])

    def Inference(self):
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256), 
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        ).to(self.device)

        model.load_state_dict(torch.load(self.weights, map_location=self.device))
        model.eval()

        sw_batch_size = 4
        roi_size = (128, 128, 64)
        with torch.no_grad():
            test_patient = first(self.test_loader)

            t_volume = test_patient['vol']
            
            test_outputs = sliding_window_inference(t_volume.to(self.device), roi_size, sw_batch_size, model)
            sigmoid_activation = Activations(sigmoid=True)
            test_outputs = sigmoid_activation(test_outputs)
            test_outputs = test_outputs > 0.25

            return Image.fromarray(np.asarray(test_outputs.detach().cpu()[0, 1, :, :, self.slice_no]))
    
        

    