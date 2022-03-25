from nilearn import datasets
from nilearn import plotting
from nilearn import image
import numpy as np

atlas_dir = "Atlas/"
#dataset = datasets.fetch_atlas_difumo(1024,resolution_mm=3,data_dir=atlas_dir)
dataset = datasets.fetch_coords_power_2011()
maps = dataset.rois
#plotting.plot_roi(atlas_filename, title="Harvard Oxford atlas",view_type='contours')
#plotting.show()

dataset = datasets.fetch_atlas_craddock_2012(atlas_dir)
atlas_filename = dataset.scorr_mean
print(image.load_img(atlas_filename).shape)
first_rsn = image.index_img(atlas_filename, 1)
#plotting.plot_roi(first_rsn, title="Cockdoc",view_type='contours')
#plotting.show()

dataset = datasets.fetch_atlas_aal(data_dir=atlas_dir)
atlas_filename = dataset.maps

plotting.plot_roi(atlas_filename, title="AAL",view_type='contours')
plotting.show()