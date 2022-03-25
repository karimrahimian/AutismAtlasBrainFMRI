from nilearn import datasets
from nilearn.datasets import fetch_abide_pcp
from nilearn.input_data import NiftiMapsMasker
from nilearn import image
from nilearn.connectome import ConnectivityMeasure
import numpy as np
from nilearn import plotting
import glob
import os

def Save_To_File(filename,corrmatrix,out_dir):
    basename = os.path.basename(filename)
    newname = basename[0:len(basename)-7]
    np.save("{}{}.npy".format(out_dir,newname),corrmatrix)

fmri_path = "data/ABIDE_pcp/newdata/*.gz"
out_dir = "Atlas_Extractor/NilearnNew/"

atlas_dir = "Atlas/"

craddock = datasets.fetch_atlas_craddock_2012(atlas_dir)
smith = datasets.fetch_atlas_smith_2009(atlas_dir)
msdl = datasets.fetch_atlas_msdl(atlas_dir)

hard = datasets.fetch_atlas_harvard_oxford('sub-prob-1mm',atlas_dir)
#mutli = datasets.fetch_atlas_basc_multiscale_2015("sym", atlas_dir)
difumo = datasets.fetch_atlas_difumo(dimension=512,resolution_mm=3,data_dir= atlas_dir)

#image = image.load_img("ROI/cc400_roi_atlas.nii/cc400.nii")

craddockmasker = NiftiMapsMasker(maps_img=craddock.scorr_mean, standardize=True, memory='nilearn_cache', verbose=0)
smithmasker70 = NiftiMapsMasker(maps_img=smith.bm70, standardize=True, memory='nilearn_cache', verbose=0)
smithmaskerr70 = NiftiMapsMasker(maps_img=smith.rsn70, standardize=True, memory='nilearn_cache', verbose=0)
hardmasker = NiftiMapsMasker(maps_img=hard.maps, standardize=True, memory='nilearn_cache', verbose=0)
difumomasker = NiftiMapsMasker(maps_img=difumo.maps, standardize=True, memory='nilearn_cache',memory_level=1, verbose=0)
msdlmasker = NiftiMapsMasker(maps_img=msdl.maps, standardize=True, memory='nilearn_cache', verbose=0)

for file in glob.glob(fmri_path):
    try:
        print(file)
        time_series = difumomasker.fit_transform(file)
        print (time_series.shape)
        correlation_measure = ConnectivityMeasure(kind='correlation')
        correlation_matrix = correlation_measure.fit_transform([time_series])[0]
        Save_To_File(file, correlation_matrix, out_dir + "difumo512/")
    except:
        print("Error in {}".format(file))

    """time_series = hardmasker.fit_transform(file)
    print (time_series.shape)
    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]
    Save_To_File(file, correlation_matrix, out_dir + "hard/")

    time_series = smithmaskerr70.fit_transform(file)
    print (time_series.shape)
    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]
    Save_To_File(file, correlation_matrix, out_dir + "rsmith70/")

    time_series = msdlmasker.fit_transform(file)
    print (time_series.shape)
    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]
    Save_To_File(file, correlation_matrix, out_dir + "msdl/")

    time_series = smithmasker70.fit_transform(file)
    print (time_series.shape)
    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]
    Save_To_File(file, correlation_matrix, out_dir + "smith70/")

    time_series = craddockmasker.fit_transform(file)
    print (time_series.shape)
    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]
    Save_To_File(file, correlation_matrix, out_dir + "craddock/")"""

#web site: https://joaoloula.github.io/functional-atlas.html