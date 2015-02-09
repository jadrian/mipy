import miraw
import miview
import numpy as np
import numpy.linalg
import os
import os.path

# Turn on interactive rendering of plots, so we can use the rawviewer functions
import matplotlib.pyplot as plt
plt.ion()


# Rather than needing to recompute the DTI, eigvecs, etc. every time, let's
# save our output to disk as we generate it.  Next time the script runs, we can
# check whether the expected file is already there, and read from disk if so.
data_root = 'dwi_data'
if not os.path.isdir(data_root):
  raise OSError(data_root + " is not a directory; please set your data_root.")

output_path = os.path.join(data_root, 'derived')
if not os.path.isdir(output_path):
  os.makedirs(output_path)


# Load the DWIs and image properties, and the gradient vectors.
# We know that this image has its dimensions ordered (X,Y,I,Z), and that the
# fastest-changing index is the first-listed (Fortran-order on disk).
DWIs_fname = os.path.join(data_root, 'dwis_cropped.float32')
(DWIs, size_info) = miraw.readRawWithSizeInfo(DWIs_fname,
                                              dimorder='XYIZ',
                                              diskorder='F')
(grad, bvals) = miraw.parseBvecs('dwi_data/bvecs.txt')
# The file doesn't specify it, but we know that our only nonzero b-value is
# 1000s/mm^2, so just load that up.
bvals = [1000] * len(bvals)


# Now that we've parsed the size_info file, we can grab volume info.
if size_info['cropped']:
  image_volume = size_info['cropped_image_size_(voxels)']
else:
  image_volume = size_info['full_image_size_(voxels)']

num_images = size_info['num_dwis']
num_unweighted = num_images - len(bvals)


# Compute the DT image.
DTI_fname = os.path.join(output_path, 'DTI.float32')
DTI_shape = image_volume + [7]
if os.path.isfile(DTI_fname):
  DTI = miraw.readRaw(DTI_fname, DTI_shape, diskorder='F')
else:
  # Compute the flattened-gradient matrix.
  # Each row is [1, -b gx^2, -b gy^2, -b gz^2, -2b gx gy, -2b gx gz, -2b gy gz].
  G = np.empty((num_images, 7), dtype=np.float32)
  for i in range(num_images):
    if i < num_unweighted:
      # For the b = 0 volume, everything except the first entry is zero.
      G[i,:] = [1, 0, 0, 0, 0, 0, 0]
    else:
      j = i-num_unweighted
      G[i,:] = [1,
                -bvals[j] * grad[j,0]**2,
                -bvals[j] * grad[j,1]**2,
                -bvals[j] * grad[j,2]**2,
                -2.0 * bvals[j] * grad[j,0] * grad[j,1],
                -2.0 * bvals[j] * grad[j,0] * grad[j,2],
                -2.0 * bvals[j] * grad[j,1] * grad[j,2]]
  # Fill the DT image
  DTI = np.empty(image_volume + [7], dtype=np.float32)
  for z in range(image_volume[2]):
    for y in range(image_volume[1]):
      for x in range(image_volume[0]):
        # This method for computing diffusion tensors does not guarantee positive-
        # definiteness: the tensor may have negative eigenvalues!  In turn, this
        # can give us negative MD or FA > 1, both of which are nonsense.
        # Visualization code should take this into account!
        sig = DWIs[x,y,:,z]
        (DTI[x,y,z,:],r,rnk,s) = np.linalg.lstsq(G, sig)
  miraw.saveRaw(DTI_fname, DTI, order='F')


# Compute eigvecs and eigvals for each tensor.
def sortedEigh(A):
  L,Q = np.linalg.eigh(A)
  I = np.argsort(-L)
  L = np.matrix(np.diag(L[I]))
  Q = Q[:,I]
  return L,Q

evecs_fname = os.path.join(output_path, 'evecs.float32')
evecs_shape = image_volume + [3,3]
evals_fname = os.path.join(output_path, 'evals.float32')
evals_shape = image_volume + [3]
if os.path.isfile(evecs_fname) and os.path.isfile(evals_fname):
  evecs = miraw.readRaw(evecs_fname, evecs_shape, diskorder='F')
  evals = miraw.readRaw(evals_fname, evals_shape, diskorder='F')
else:
  evecs = np.empty(evecs_shape, dtype=np.float32)
  evals = np.empty(evals_shape, dtype=np.float32)
  for z in range(image_volume[2]):
    for y in range(image_volume[1]):
      for x in range(image_volume[0]):
        D = np.empty((3,3), dtype=np.float32)
        D[0,0] = DTI[x,y,z,1]
        D[1,1] = DTI[x,y,z,2]
        D[2,2] = DTI[x,y,z,3]
        D[0,1] = DTI[x,y,z,4]
        D[1,0] = DTI[x,y,z,4]
        D[0,2] = DTI[x,y,z,5]
        D[2,0] = DTI[x,y,z,5]
        D[1,2] = DTI[x,y,z,6]
        D[2,1] = DTI[x,y,z,6]
        (Lambda,Q) = sortedEigh(D)
        evecs[x,y,z,:,:] = Q
        evals[x,y,z,:] = np.diag(Lambda)
  miraw.saveRaw(evecs_fname, evecs, order='F')
  miraw.saveRaw(evals_fname, evals, order='F')


# Compute FA and MD.
FA_fname = os.path.join(output_path, 'FA.float32')
MD_fname = os.path.join(output_path, 'MD.float32')
if os.path.isfile(FA_fname) and os.path.isfile(MD_fname):
  FA = miraw.readRaw(FA_fname, image_volume, diskorder='F')
  MD = miraw.readRaw(MD_fname, image_volume, diskorder='F')
else:
  MD = np.empty(image_volume, dtype=np.float32)
  FA = np.empty(image_volume, dtype=np.float32)
  for z in range(image_volume[2]):
    for y in range(image_volume[1]):
      for x in range(image_volume[0]):
        MD[x,y,z] = np.sum(evals[x,y,z,:]) / 3.0
        l = evals[x,y,z,:]
        FA_num = ((l[0]-l[1])*(l[0]-l[1]) +
                  (l[0]-l[2])*(l[0]-l[2]) +
                  (l[1]-l[2])*(l[1]-l[2]))
        FA_den = np.dot(l,l)
        FA[x,y,z] = np.sqrt(1.5 * FA_num / FA_den)
  miraw.saveRaw(FA_fname, FA, order='F')
  miraw.saveRaw(MD_fname, MD, order='F')


# Render a color image of principal eigenvector orientation.
orient_color = np.empty(image_volume + [3], dtype=np.float32)
for z in range(image_volume[2]):
  for y in range(image_volume[1]):
    for x in range(image_volume[0]):
      # Compute color values from principal eigvec components.
      # Remember FA can be >1, since we do a messy job fitting the tensors;
      # thus we clamp our FA scaling to 1.
      cur_color = np.abs(evecs[x,y,z,:,0]) * min(1, FA[x,y,z])
      orient_color[x,y,z,:] = cur_color

# Now we can call, e.g., miview.stackToVideo(orient_color)
