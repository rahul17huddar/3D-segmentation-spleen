import streamlit as st
import matplotlib.pyplot as plt
from testing import MONAI
import os
import torch
from glob import glob
import tempfile

device = torch.device('cpu')

st.title('3D Medical Image Segmentation',)
st.header('Spleen Segmentation in Chest CT Scans using Computer Vision')
st.write("""The spleen is located in the upper left part of the belly under the ribcage. 
         It helps protect the body by clearing worn-out red blood cells and other foreign bodies (such as germs) from the bloodstream.
         Since Spleen is involved in so many bodily functions, it is vulnerable to a range of disorders. 
         Locating the spleen in a CT scan is important for treatments of spleen disorders.""")
st.image('spleen.png')
st.write("""3D segmentation is the process of classifying different structures or regions within a 3D volume, typically derived from medical imaging data such as MRI or CT scans.
         Nifti(Neuroimaging Informatics Technology Initiative) is a medical image format, to store both images, and companied data, the images are usually in grayscale, and 
         they are taken as slices, each slice with a different cross-section of the body. A typical Nifti file has 3 dimensions i.e (Image height, Image width, Number of Slices) """)
st.write("""Here you can upload a chest CT scan file(nifti file) to find the location of the spleen in it or you can try out the "Use default Nifti Image" to know how it works. """)
uploaded_file = st.file_uploader("Upload a nifti file", ["nii.gz"])
st.write('Note: Only upload file with .nii.gz extension')
st.write('OR')
use_default_images = st.checkbox('Use default Nifti Image')

if use_default_images:
        nifti_no = st.selectbox('You can choose any 1 of the 8 CT scans of 8 diffrent patients', (1,2,3,4,5,6,7,8))
        st.write('You selected nifti file number:',nifti_no)
        st.write("There are 64 slices in this nifti file, slide the slider below to view each slice. When spleen is detected in any of the slice, you'll see a white mask on spleen and black mask for background in the output image. If there is no spleen in the slice the output will be black mask.")
        slice_no = st.slider('Slide to choose the slice number', min_value=1, max_value=64)
        st.write('You selected slice number:',slice_no)
        in_dir = "Data_Spleen"
        with st.spinner('Niffti Image is loading'):
            test_images = sorted(glob(os.path.join(in_dir, "TestVolumes", "*.nii.gz")))
            test_segmentation = sorted(glob(os.path.join(in_dir, "TestSegmentation", "*.nii.gz")))
            test_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in zip(test_images, test_segmentation)]
            test_files = test_files[nifti_no:10]

            instance = MONAI('best_metric_model.pth', device, test_files, slice_no)

            orignal_image = instance.ViewImage()
            output = instance.Inference()
            images = [orignal_image, output]
            
            col1, col2 = st.columns(2)

            col1.image(images[0], width=300, caption='Image')
            col2.image(images[1], width=300, caption='Output')

elif uploaded_file is not None:
    temp_dir = tempfile.TemporaryDirectory()
    file_name = uploaded_file.name
    file_extension = os.path.splitext(file_name)[1]
    
    file_path = os.path.join(temp_dir.name, file_name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    
    st.success(f"Successfully uploaded '{file_name}")
    test_files = [{"vol": file_path}]
    st.write("There are 64 slices in this nifti file, slide the slider below to view each slice. When spleen is detected in any of the slice, you'll see a white mask on spleen and black mask for background in the output image. If there is no spleen in the slice the output will be black mask.")
    slice_no = st.slider('Slide to choose the slice number', min_value=1, max_value=64)
    st.write('You selected a slice number:',slice_no)
    with st.spinner('Niffti Image is loading'):
        instance = MONAI('best_metric_model.pth', device, test_files, slice_no)

        orignal_image = instance.ViewImage()
        output = instance.Inference()
        images = [orignal_image, output]
        
        col1, col2 = st.columns(2)

        col1.image(images[0], width=300, caption='Image')
        col2.image(images[1], width=300, caption='Output')
            