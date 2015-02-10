import numpy as np
import os.path
import mipy._miraw_helpers as _miraw_helpers
try:
  from nibabel import nifti1
  has_nifti = True
except ImportError:
  import warnings
  w = """You don't have nibabel installed, so reading and writing nifti files
won't be supported for you.  nibabel exists for both Python 2 and 3, though,
so look into installing it."""
  warnings.warn(w, RuntimeWarning)
  has_nifti = False

"""Functions for dealing with raw medical imaging datasets.

This module is particularly focused on working with diffusion-weighted images
and derived images, which are typically 4-D (3 for space, plus one dimension for
arbitrary sample vectors).  Its default metadata format, "size_info", is a hacky
thing custom-built just for DWIs, and does not accommodate N-D volumes for N < 3
or N > 4.  The readRaw() and saveRaw() functions, however, will work for any N.
"""

def readRaw(f, shape, dtype=None, diskorder='F', memorder='C'):
  """Loads array data from a raw binary file on disk.
  
  This is a wrapper around numpy.fromfile, and returns a numpy.ndarray that
  owns its own memory.  Its particular purpose is to work with differing
  dimension orderings on disk and in memory.  The default is to interpret
  the file as "Fortran-ordered" (Matlab's default; column-major; first index
  is fastest-changing) and to produce an ndarray that is "C-ordered" (numpy's
  default; row-major; last index is fastest-changing).
  
  This function does not support memory mapping (yet), so it's not
  appropriate to use if your array is too big to fit comfortably in memory.
  numpy.load() and the h5py package are alternatives, but they put
  restrictions on the file format.  numpy.memmap() may be the right option.
  
  Arguments:
    f:         An open file object or a filename.
    shape:     A tuple of dimension extents.  One dimension may be given
               extent -1; if so, this dimension stretches to fit all the
               voxel values.
    dtype:     A numpy data type (like numpy.float32).  If None, dtype is
               inferred from the filename.
    diskorder: 'F' or 'C', default 'F'.
    memorder:  'F' or 'C', default 'C'.
  
  Throws a ValueError if shape does not match the number of voxels stored on
  disk, or if the product of the non-negative values in shape does not divide
  the number of voxels evenly.
  
  Returns a numpy.ndarray with the given shape and order=memorder.
  """
  # Read the data into a flat array.
  if dtype is None:
    dtype = _miraw_helpers.inferDtypeFromFilename(f)
  raw = np.fromfile(f, dtype=dtype)
  
  # Resolve the shape argument.
  shape = np.array(shape)
  num_voxels = np.prod(shape)
  if num_voxels < 0:
    num_voxels = -num_voxels
    missing_dim = int(raw.shape[0] / num_voxels)
    if num_voxels * missing_dim != raw.shape[0]:
      err = (('File has %i voxels; you gave me shape = %s = %i voxels,\n' +
              'which does not divide evenly.') %
             (raw.shape[0], repr(shape.tolist()), num_voxels))
      raise ValueError(err)
    # Replace the missing dimension.
    shape = np.where(shape < 0, missing_dim, shape)
  
  # Reshape the flat array, interpreting according to the disk order.
  try:
    X = np.ndarray(shape=shape, dtype=dtype, buffer=raw.data, order=diskorder)
  except TypeError:
    num_voxels = np.prod(shape)
    if num_voxels != raw.shape[0]:
      err = ('File has %i voxels; you gave me shape = %s = %i voxels.' %
             (raw.shape[0], repr(shape.tolist()), num_voxels))
      raise ValueError(err)
    else:
      raise
  
  # Now convert to the memory order and return.
  return _miraw_helpers.ndcopyWithOrder(X, memorder)


def saveRaw(f, X, order='F', dtype_as_ext=False):
  """Save array data to a raw binary file on disk.
  
  This is a wrapper around numpy.ndarray.tofile.  Its particular purpose is
  to enable the creation of "Fortran-ordered" raw files, (aka column-major;
  Matlab's default), in which the fastest-changing index in the source array,
  with respect to the linear order in which data are stored on disk, is the
  first index, rather than the last index ("C-ordered", numpy's default).
  
  Arguments:
    f:            An open file object or a filename.
    X:            A numpy ndarray, with any shape and storage order.
    order:        'F' or 'C' --- the order for storage on disk.
    dtype_as_ext: If "True" and f is a string, appends the dtype as an
                  extensions for the filename.  Raises TypeError if True and
                  f is not a string.
  """
  if dtype_as_ext:
    if isinstance(f, str):
      f += '.' + str(X.dtype)
    else:
      raise TypeError("Can't append extension to an open file object.")
  X.flatten(order=order).tofile(f)


def readRawWithSizeInfo(f, sizefile=None, dtype=None, cropped=None, dimorder=None, diskorder='F', memorder='C'):
  """Loads a raw image file from disk, using a size_info metadata file.
  
  Arguments:
    f:         A filename or open file object for the raw data file.
    sizefile:  A filename for the size_info metadata file.
               If None, looks for a file called "size_info" in f's directory.
    dtype:     The numpy dtype of the raw data, or None.
    cropped:   A boolean: whether f is a cropped or full volume (as described in
               the sizefile), or None (in which case this will be inferred).
    dimorder:  Four-character string that is a permutation of "XYZI",
               indicating the dimension order of the image being read from
               disk.
                 The purpose of this argument is to map from the dimension
               extents stored in the size_info file, which are always stored
               in XYZI order, to the actual shape of the ndarray we create.
               Namely, if we create a 4-tuple "dimmap" by converting each
               character X->0, Y->1, Z->2, I->3, then
                   vol.shape[i] = sz[dimmap[i]]
               for i from 0 to 3, where vol is the returned volume and sz is
               the volume size, in (X,Y,Z,I) order, read from size_info.
                 The default value, None, is equivalent to "XYZI".
                 This can be a confusing argument, so please note:
                 - dimorder is overridden if the size_info file specifies a
                   "dimension_order" value.
                 - dimorder only indicates a rearrangement of dimension
                   extents from the default order (as read in the size_info
                   file) to the order that dictates the shape attribute of the
                   returned array.  Though it interacts in complicated ways
                   with the diskorder and memorder arguments, ultimately it is
                   not equivalent to calling transpose(dimmap) on the returned
                   array.
                 - dimorder does not change the order of the dimension extents
                   as stored in the returned dictionary "cfg".
    diskorder: The array traversal order of the file.
    memorder:  The desired traversal order of the output array.
  
  (See readRaw for more explanation of the last two arguments.)
  
  This function attempts, usually successfully, to infer the values of
  arguments left None.
  
  Returns (vol, cfg), where vol is a numpy ndarray and cfg is the dict of
  settings in sizefile.  In addition, this function defines an additional
  key, cfg['cropped'], with Boolean value.
  """
  
  # Read the image into a 1-D array.
  raw = readRaw(f, (-1,1), dtype=dtype, diskorder=diskorder, memorder=memorder)
  
  # Read the size file.
  imgname = _miraw_helpers.getFilename(f)
  if sizefile is None:
    try:
      sizefile = os.path.join(os.path.dirname(imgname), 'size_info')
    except:
      raise TypeError("Can't infer sizefile from filename '%s'." % imgname)
  cfg = readConfigFile(sizefile)
  sz = cfg['full_image_size_(voxels)']
  sz_c = cfg['cropped_image_size_(voxels)']
  try:
    n_imgs = cfg['num_dwis']
  except KeyError:
    n_imgs = 1
  
  # Try to figure out whether the image is cropped.
  cropped, threeD = _miraw_helpers.detectShapeAndCropping(raw.size,
    np.prod(sz), np.prod(sz_c), n_imgs, cropped)
  if cropped:
    sz = sz_c
  sz = sz + [n_imgs]
  cfg['cropped'] = cropped
  
  # Finally set the size and return.
  try:
    dimorder = cfg['dimension_order']
  except KeyError:
    if dimorder is None:
      dimorder = _miraw_helpers.DIM_DEFAULT_ORDER
  if not _miraw_helpers.isValidDimorder(dimorder):
    raise ValueError('"%s" is not a valid dimorder argument.' % repr(dimorder))
  
  if threeD:
    sz = sz[0:3]
  else:
    sz = np.take(sz, _miraw_helpers.dimorderToDimmap(dimorder), axis=0)
  return (_miraw_helpers.ndcopyWithOrder(raw.reshape(sz, order=diskorder),
                                         memorder),
          cfg)


def saveSizeInfo(f, img, vox_sz=(1,1,1), dimorder=None, size_cfg={}, infer_name=False):
  """Write a size_info metadata file to disk for a given array.
  
  A size_info file stores image (array) dimensions for raw images, as well as
  voxel size and cropping information (indicating that the array is cropped
  from a larger volume).  Note that size_info is designed for 3-D or 4-D
  arrays only, and stores the extents of the first three dimensions
  separately from that of the fourth.
  
  Arguments:
    f:          An open file handle or a filename for the destination file.
    img:        A numpy.ndarray.
    vox_sz:     Optional array-like object with 2 or 3 entries.
    dimorder:   Four-character string that is a permutation of "XYZI",
                indicating the dimension order of the image "img".
                  The purpose of this argument is to map from the dimension
                extents represented in img.shape to the extents stored in the
                size_info file, which are always stored in a canonical "XYZI"
                order.  Namely, if we create a 4-tuple "dimmap" by converting
                each character X->0, Y->1, Z->2, I->3, then
                   sz[dimmap[i]] = img.shape[i]
                for i from 0 to 3, where sz is the volume size, in (X,Y,Z,I)
                order, that will be stored in the size_info.
                  The default value, None, is equivalent to "XYZI".
    size_cfg:   Optional dictionary of other config key-value pairs.  The data
                stored in this dictionary override all values computed by or
                passed into this function, even if they're inconsistent with
                the size of the image.  Be careful!  This includes the
                "dimension_order" value, which overrides the dimorder argument
                above.
    infer_name: Optional boolean.  If True, and f is a filename, then the
                file actually written will be in the same directory as f
                (or in f if f is a path ending with a slash), and named
                "size_info".  If f is not a string, this option has no effect.
  """
  # Deal with filenames and open a file for writing, if necessary.
  if isinstance(f, str):
    if infer_name:
      f = os.path.join(os.path.dirname(f), 'size_info')
    fid = open(f, 'w')
    close_after = True
  else:
    fid = f
    close_after = False
  
  # Set up dimension mapping.
  shape = list(img.shape) + [1]*4
  try:
    dimorder = size_cfg['dimension_order']
  except KeyError:
    if dimorder is None:
      dimorder = _miraw_helpers.DIM_DEFAULT_ORDER
  if not _miraw_helpers.isValidDimorder(dimorder):
    raise ValueError('"%s" is not a valid dimorder argument.' % repr(dimorder))
  shape = np.take(shape, _miraw_helpers.dimorderToReverseDimmap(dimorder), axis=0).tolist()
  
  # Extract metadata from the arguments.
  base_keys = ['voxel_size_(mm)', 'full_image_size_(voxels)',
               'low_end_crop_(voxels)', 'cropped_image_size_(voxels)',
               'num_dwis', 'dimension_order']
  auto_cfg = {
    base_keys[0] : vox_sz,
    base_keys[1] : shape[:3],
    base_keys[2] : [0, 0, 0],
    base_keys[3] : shape[:3],
    base_keys[4] : shape[3],
    base_keys[5] : dimorder
  }
  
  # Overwrite metadata with what the user gave us.
  for (k,v) in size_cfg.items():
    auto_cfg[k] = v
  
  # Now write the key-value pairs and we're done!
  def spaceSepStr(a):
    if isinstance(a, list) or isinstance(a, tuple):
      return ' '.join([str(x) for x in a])
    return str(a)
  
  for k in base_keys:
    fid.write(k + ': ' + spaceSepStr(auto_cfg[k]) + '\n')
    del(auto_cfg[k])
  for (k, v) in auto_cfg.items():
    fid.write(k + ': ' + spaceSepStr(v) + '\n')
  
  if close_after:
    fid.close()


def applyDimOrder(img, dimorder):
  """Permutes the data dimensions of img as specified by dimorder.
  
  Arguments:
    img:      numpy ndarray with four dimensions.
    dimorder: Four-character string that is a permutation of "XYZI",
              indicating the desired dimension order of the output.
  
  Preconditions:
    - The current dimension order of "img" is "XYZI".
  
  Returns new_img:
    new_img:  numpy ndarray with four dimensions.  The values in new_img will
              be rearranged so that the dimension order is as specified by
              dimorder.
  
  Note that if you have any metadata about the image, like voxel sizes, the
  order of values in your metadata will no longer match the order of the
  dimensions in the output image.  You'll need to rearrange those manually with
  applyDimOrderToList().
  """
  return img.transpose(_miraw_helpers.dimorderToDimmap(dimorder))


def applyDimOrderToList(L, dimorder):
  """Permutes the values in L as specified by dimorder.
  
  Arguments:
    L:        A list of four values, corresponding (respectively) to the
              X, Y, Z, and I dimensions of some dataset.
                If you've only got values for X, Y, and Z, pad before calling.
    dimorder: Four-character string that is a permutation of "XYZI",
              indicating the desired dimension order of the output.
  
  Returns a permuted version of L.
  """
  return [L[i] for i in _miraw_helpers.dimorderToDimmap(dimorder)]


def undoDimOrder(img, dimorder):
  """Permutes the data dimensions of img, which currently has the given
  dimension order, to match the default "XYZI" dimension order.
  
  Arguments:
    img:      numpy ndarray with four dimensions.
    dimorder: Four-character string that is a permutation of "XYZI",
              indicating the current dimension order of img.
  
  Returns new_img:
    new_img:  numpy ndarray with four dimensions.  The values in new_img will
              be rearranged so that the dimension order is XYZI.
  """
  return img.transpose(_miraw_helpers.dimorderToReverseDimmap(dimorder))


def undoDimOrderOnList(L, dimorder):
  """Permutes the values in L to restore them to a default dimorder.
  
  Arguments:
    L:        A list of four values, corresponding (in order) to the dimensions
              of some dataset.
    dimorder: Four-character string that is a permutation of "XYZI",
              indicating the current dimension order of the dataset.
  
  Returns a permuted version of L, with values corresponding (respectively) to
  the X, Y, Z, and I.
  """
  return [L[i] for i in _miraw_helpers.dimorderToReverseDimmap(dimorder)]


def parseBvecs(f):
  """Parse a b vector file's contents.
  
  Arguments:
    f: An open file handle, a filename string, or a string of the contents of
       a b-vector file.
  
  Prerequisites: this function supports two different plaintext formats
    for describing a list of three-element vectors (optionally, with an
    additional scalar associated with each vector):
      A: three (or four) lines of N space- or comma-delimited values each
      B: N lines of three or four space- or comma-delimited values each
    Comments (with #, //, or %) are okay, as are blank lines.
  
  Returns a 2-tuple, (vecs, b):
    vecs: a numpy Nx3 array, each row being a 3-vector (of length 1 or 0)
    b:    an array of length N, each element of which is a float.  If no
          b-values are found in the file, then each element is None.
  """
  string = ''
  if hasattr(f, 'read'):
    string = f.read()
  elif isinstance(f, str):
    if f.find('\n') < 0:
      # f is a filename
      fid = open(f, 'r')
      string = fid.read()
      fid.close()
    else:
      string = f
  else:
    raise TypeError('f argument must be either a string or a file object.')
  
  lines = _miraw_helpers.cleanlines(string.splitlines())
  vecs = np.array([[float(y) for y in x.replace(',',' ').split()] for x in lines])
  
  if vecs.shape[0] <= 4:
    # Format A: transpose to make it match format B.
    vecs = vecs.T
  if vecs.shape[1] < 3 or vecs.shape[1] > 4:
    raise IndexError('Vectors must each have three components.')
  if vecs.shape[1] == 4:
    # Separate out the b-values.
    b = vecs[:,3]
    vecs = lines[:,0:3]
  else:
    b = [None] * vecs.shape[0]
  
  # Normalize the vectors: sum the squares along each row, then divide by
  # nonzero ones.
  norms = np.array(np.sqrt(np.sum(vecs**2, axis=1)))
  norms = np.where(norms < 1e-6, 1, norms)
  vecs = vecs / norms[:, np.newaxis]
  
  return (vecs, b)


if has_nifti:
  def readNifti(infile, dtype=None, memorder=None):
    """Read a NIfTI file into a numpy array.
    
    Arguments:
      infile:    Filename of a NIfTI file, or a list of strings.  The list
                 indicates a sequence of files to be concatenated together,
                 in the order given, in the I dimension (the 4th dimension).
      dtype:     A numpy data type to cast to.  If None, the data type remains
                 whatever the NIfTI header specifies.
      memorder:  'F' or 'C' --- the order for storage in memory.  If None, the
                 order remains whatever the NIfTI library read it as (most
                 likely 'F', the standard layout order for NIfTI).
    
    Returns (vol, cfg, header):
      vol:    numpy ndarray.
      cfg:    dict storing information that you would find in a size_info file.
      header: the NIfTI header of infile (or of the first-listed file, with the
              fourth dimension size changed).
    
    Note that if the NIfTI header encodes a dimension flip or exchange, this
    function DOES NOT apply it to the image before returning.  You'll want to
    check that with niftiGetXform() and perhaps fix it with applyNiftiXform().
      If you ultimately want a raw volume with a non-standard dimension order,
    you should apply that AFTER you apply the NIfTI transform, since
    applyNiftiXform() assumes that the dimension order is precisely as
    represented in the original raw data from the NIfTI file.
      Here's the recommended procedure: read, apply the transform, and then
    remap to the desired dimension order:
        (vol, cfg, header) = readNifti(fname)
        (vol, xform, vox_sz) = applyNiftiXform(vol, niftiGetXform(header), cfg)
        vol = applyDimOrder(vol, dimorder)
      This procedure is what niftiToRaw() does.
    """
    if isinstance(infile, str):
      # Read this one file.
      nii = nifti1.load(infile)
      raw = nii.get_data()
      header = nii.header
    elif isinstance(infile, list):
      # Read a list of files: first read in file 0...
      nii = nifti1.load(infile[0])
      raw = nii.get_data()
      header = nii.header
      raw.resize(raw.shape + (1,)*(4-raw.ndim))
      # ... then concatenate on each other one.
      for i in range(1, len(infile)):
        nii = nifti1.load(infile[i])
        newraw = nii.get_data()
        newraw.resize(newraw.shape + (1,)*(4-newraw.ndim))
        raw = np.concatenate((raw, newraw), axis=3)
      header.set_data_shape(raw.shape)
    else:
      raise ValueError('"%s" is not a valid infile argument.' % repr(infile))
    
    curr_dtype = raw.dtype
    if np.isfortran(raw):
      curr_memorder = "F"
    else:
      curr_memorder = "C"
    
    if dtype is None:
      dtype = curr_dtype
    if memorder is None:
      memorder = curr_memorder
    
    # Create the size_info config dict.
    cfg = {}
    cfg['voxel_size_(mm)']             = header['pixdim'][1:4].tolist()
    cfg['full_image_size_(voxels)']    = raw.shape[:3]
    cfg['low_end_crop_(voxels)']       = [0,0,0]
    cfg['cropped_image_size_(voxels)'] = cfg['full_image_size_(voxels)']
    if len(raw.shape) > 3:
      cfg['num_dwis']                  = raw.shape[3]
    else:
      cfg['num_dwis']                  = 1
    cfg['dimension_order']             = _miraw_helpers.DIM_DEFAULT_ORDER
    
    return (raw.astype(dtype, order=memorder), cfg, header)
  
  
  def applyNiftiXform(img, xform, cfg=None):
    """Flips and exchanges dimensions in the given raw image according to a
    NIfTI-style 4x4 transform matrix.  The resulting image should conform to
    the NIfTI standard interpretation: the fastest-changing index is X, going
    left to right; the next-fastest is Y, going posterior to anterior, and the
    slowest is Z, going inferior to superior.
    
    Arguments:
      img:   numpy.ndarray with at least three dimensions.
      xform: 4x4 numpy array.  xform[:3][:3] must contain exactly three nonzero
             entries, one on each row.
      cfg:   Optional dictionary of image metadata.
    
    Returns (new_img, new_xform, vox_sz, cfg):
      new_img:   Transformed image.
      new_xform: Modified transform.  new_xform[:3,:3] is diagonal with positive
                 entries.
      vox_sz:    Length-3 numpy vector of positive voxel sizes.
    
    If cfg is provided, this function overwrites the values for the following
    keys:
      'voxel_size_(mm)'             -> Voxel size with new dimension order.
      'full_image_size_(voxels)'    -> Image size with new dimension order.
      'low_end_crop_(voxels)'       -> [0,0,0]
      'cropped_image_size_(voxels)' -> (Same as full image size)
      'num_dwis'                    -> Size of 4th dimension (shouldn't change).
      'dimension_order'             -> "XYZI"
    
    In the case of a transform that has an oblique rotation or affine component,
    this function raises a ValueError.
    """
    # According to the NIfTI spec*, the xform applies by left-multiplication to
    # the column vector [i,j,k,1]' to specify how spatial coordinates [x,y,z]'
    # may be computed from the raw image indices:
    #   [ s_x[0], s_x[1], s_x[2], s_x[3] ]   [ i ]   [ x ]
    #   [ s_y[0], s_y[1], s_y[2], s_y[3] ] * [ j ] = [ y ]
    #   [ s_z[0], s_z[1], s_z[2], s_z[3] ]   [ k ]   [ z ]
    #                                        [ 1 ]
    # For example, this matrix
    #   [  0 -3  0  0 ]
    #   [  0  0  5  0 ]
    #   [ -4  0  0  0 ]
    # means that
    #   j encodes the x direction, with an x-flip and a voxel size of 3
    #   k encodes the y direction, with no y-flip and a voxel size of 5
    #   i encodes the z direction, with a  z-flip and a voxel size of 4
    # In other words, the dimension order is ZXY.
    # 
    # * http://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/srow.html
    
    if xform.shape != (4,4):
      raise ValueError('xform must be 4x4.')
    
    # --- First, exchange dimensions as specified. ---
    # Turn all nonzero entries in the upper part to ones.
    perm = np.array(np.abs(xform[:3,:3]) > 1e-6, dtype=int)
    dim_map = np.nonzero(perm)[1]
    # dim_map[d] says which column of perm is nonzero in row d.  In context,
    # dim_map[d] indicates which volume index encodes direction d.  For example,
    # dim_map = [1,2,0] means that j (index=1) encodes x (d=0), k encodes y, and
    # i encodes z.
    
    if dim_map.size != 3:
      # There should be exactly three nonzero entries in the upper-left 3x3
      # portion of the transform matrix.
      # TODO Add support for affine transforms.
      raise ValueError('Bad transform --- too many nonzero entries.\n%s', repr(xform))
    
    # Tack on extra un-permuted entries for any non-spatial dimensions.
    dim_map = dim_map.tolist() + list(range(3,img.ndim))
    
    # N-D transpose so that direction d is encoded by index d, for all d.
    new_img = img.transpose(dim_map)
    
    # Permute the xform matrix to match.
    perm = np.pad(perm, ((0,1),(0,1)), mode='constant')
    perm[3,3] = 1
    new_xform = xform.dot(perm.transpose())
    
    # The result here should be:
    #  - new_img has had its dimensions rearranged appropriately.
    #  - new_xform[:3,:3] is a diagonal matrix of (possibly negative) voxel sizes.
    off_diag = np.nonzero((np.ones((3,3))-np.eye(3))*new_xform[:3,:3])[0].size
    if off_diag != 0:
      raise ValueError('The transformation failed.  This should never happen!')
    
    # --- Now flip axes as specified. ---
    vox_sz = np.diag(new_xform[:3,:3]).tolist()
    for d in range(3):
      if vox_sz[d] < 0.0:
        # To reverse a specified dimension, we need to swap it with dimension
        # zero, flip it, and then swap back.  Further explanation:
        # http://stackoverflow.com/questions/13240117/reverse-an-arbitrary-dimension-in-an-ndarray
        new_img = new_img.swapaxes(d, 0)
        new_img = new_img[::-1, ...]
        vox_sz[d] *= -1.0
        new_xform[d,d] *= -1.0
        new_xform[d,3] = new_xform[d,3] - (new_img.shape[0] * vox_sz[d])
        new_img = new_img.swapaxes(0, d)
    
    if cfg is not None:
      cfg['voxel_size_(mm)']             = vox_sz
      cfg['full_image_size_(voxels)']    = new_img.shape[:3]
      cfg['low_end_crop_(voxels)']       = [0,0,0]
      cfg['cropped_image_size_(voxels)'] = new_img.shape[:3]
      cfg['num_dwis']                    = new_img.shape[3]
      cfg['dimension_order']             = _miraw_helpers.DIM_DEFAULT_ORDER
    
    return (new_img, new_xform, vox_sz)
  
  
  def niftiGetXform(hdr):
    """Extracts a single 4x4 transform matrix from a NIfTI header object.
    """
    (qform, qcode) = hdr.get_qform(True)
    (sform, scode) = hdr.get_sform(True)
    if qcode + scode == 0:
      # Neither gave us an answer.
      return np.eye(4)
    elif scode == 1:
      # We prefer the sform, since it can represent an affine matrix, so we
      # return it even if qform is also defined.
      return sform
    else:
      return qform
  
  
  def niftiToRaw(infile, outfile=None, sizefile=None, dimorder=None, diskorder='F', dtype=None, dtype_as_ext=False):
    """Convert a NIfTI file (or set of files) to a raw file.
    
    Arguments:
      infile:       Filename of a NIfTI file, or a list of strings.  The list
                    indicates a sequence of files to be concatenated together,
                    in the order given, along dimension 4 (where 1 is the
                    fastest-changing).
      outfile:      Filename of a raw file to generate.  If None, the filename
                    will be copied from infile, but with an extension indicating
                    the dtype.  See also dtype_as_ext.
      sizefile:     Filename of a size_info metadata file.  If None, it will go
                    in the same directory as outfile.  If empty string, no
                    size_info file will be generated.
      dimorder:     Four-character string that is a permutation of "XYZI",
                    indicating the desired dimension order of the output image.
                      The default value, None, is equivalent to "XYZI".
      diskorder:    'F' or 'C' --- the order for storage on disk.
      dtype:        A numpy data type to cast to.  If None, the data type
                    either remains whatever the NIfTI header specifies, or is
                    cast to the type specified by the extension on outfile.
      dtype_as_ext: If True, and if outfile is not None, then this appends the
                    dtype to the end of outfile.
    """
    if dimorder is None:
      dimorder = _miraw_helpers.DIM_DEFAULT_ORDER
    if not _miraw_helpers.isValidDimorder(dimorder):
      raise ValueError('"%s" is not a valid dimorder argument.' % repr(dimorder))
    
    # Figure out the desired dtype.
    fname_dtype = None
    try:
      fname_dtype = _miraw_helpers.inferDtypeFromFilename(outfile)
    except:
      pass
    
    if dtype is not None and fname_dtype is not None:
      if fname_dtype != np.dtype(dtype):
        raise ValueError("Arguments specify contradictory dtypes:\n  outfile: {}\n  dtype:   {}".format(outfile, dtype))
    elif dtype is None:
      dtype = fname_dtype
    
    # Now either dtype is None, because both the outfile and dtype arguments
    # failed to set it, or it and the outfile agree.  If it's None, then we'll
    # just keep the dtype from the NIfTI file.
    
    # Read the file and set the dtype once and for all.
    (img, cfg, header) = readNifti(infile, dtype)
    dtype = img.dtype
    
    # Apply any dimension flips or permutations according to the header.
    (img, xform, vox_sz) = applyNiftiXform(img, niftiGetXform(header), cfg)
    
    # And finally put the data in the requested storage order.
    img = applyDimOrder(img, dimorder)
    
    # Generate new names for the output files as necessary.
    if outfile is None:
      if not isinstance(infile, str):
        raise ValueError("No outfile specified, but infile %s is not a string!" % repr(infile))
      (base, ext) = os.path.splitext(infile)
      if ext == ".gz":
        (base, ext) = os.path.splitext(base)
      outfile = base + "." + str(dtype)
    elif dtype_as_ext:
      outfile += "." + str(dtype)
    
    if sizefile is None:
      sizefile = os.path.join(os.path.dirname(outfile), "size_info")
    
    # Write the size_info file.
    if len(sizefile) > 0:
      saveSizeInfo(sizefile, img, size_cfg=cfg, infer_name=False)
    
    # And finally write the raw file.
    saveRaw(outfile, img, diskorder, dtype_as_ext)
  
  
  def rawToNifti(infile, sizefile=None, outfile=None, dimorder=None, diskorder='F', dtype=None, split4=False):
    """Convert a raw file to a NIfTI file.
    
    Arguments:
      infile:    Filename of a raw file.
      sizefile:  Filename of a size_info config file.  If None, attempts to find
                 this file in the same directory as infile.
      outfile:   Filename (including .nii) of the NIfTI file to generate.
                 If None, it will be generated from infile.
      dimorder:  Four-character string that is a permutation of "XYZI",
                 indicating the dimension order of the image in "infile".
                   The purpose of this argument is to rearrange the order of the
                 dimensions in the infile to match the NIfTI canonical order of
                 (X, Y, Z, I), where I is the dimension along which multiple
                 acquisitions are concatenated.
                   The default value, None, is equivalent to "XYZI".
                   Note that this argument will be overridden if the size_info
                 file contains a "dimension_order" value.
      diskorder: A string, 'F' or 'C', representing the order in which the data
                 values are stored in the raw file.
      dtype:     The numpy dtype for the infile.  If None, it is inferred from
                 infile's extension.
      split4:    If True, output numbered 3-D images from 4-D input.
    """
    (raw, cfg) = readRawWithSizeInfo(infile, sizefile=sizefile, dtype=dtype,
                                     dimorder=dimorder, diskorder=diskorder,
                                     memorder='C')
    vox_sz = cfg['voxel_size_(mm)']
    
    # Rearrange dimensions.
    try:
      dimorder = cfg['dimension_order']
    except KeyError:
      if dimorder is None:
        dimorder = _miraw_helpers.DIM_DEFAULT_ORDER
    if not _miraw_helpers.isValidDimorder(dimorder):
      raise ValueError('"%s" is not a valid dimorder argument.'%repr(dimorder))
    raw_transp = raw.transpose(_miraw_helpers.dimorderToReverseDimmap(dimorder))
    
    if split4 and len(raw_transp.shape) == 4:
      raw_stack = [raw_transp[:,:,:,i] for i in range(raw_transp.shape[3])]
    else:
      raw_stack = [raw_transp]
    i = 0
    for img in raw_stack:
      nii = nifti1.Nifti1Pair(img, np.diag(vox_sz + [0.0]))
      nii.get_header().set_xyzt_units('mm')
      outfname = outfile
      if outfname is None:
        outfname = os.path.splitext(infile)[0] + '.nii'
      if split4:
        outfname = os.path.splitext(outfname)[0] + ('_%03i.nii' % i)
        i += 1
      nifti1.save(nii, outfname)


def parseConfig(s):
  """Parse a simple config file.
  
  The expected format encodes a simple key-value store: keys are strings,
  one per line, and values are arrays.  Keys may not have colons in them;
  everything before the first colon on each line is taken to be the key,
  and everything after is considered a space-separated list of value-array
  entries.  Leading and trailing whitespace are stripped on each key and
  value entry.
  
  No special handling of comments is implemented, but non-conforming lines
  (those with no colon) will be silently ignored.
  
  Arguments:
    s: A string containing the full contents of a config file.
  
  Returns a dictionary mapping strings to lists.  The lists, which may be
  singletons, contain ints, floats, and/or strings.
  """
  def stringToNumberMaybe(s):
    if s.lower() in ['true', 'yes']:
      return True
    if s.lower() in ['false', 'no']:
      return False
    try:
      return int(s)
    except ValueError:
      try:
        return float(s)
      except ValueError:
        return s
  
  lines = s.splitlines()
  d = {}
  for line in lines:
    kv = [x.strip() for x in line.split(':',1)]
    try:
      val_list = [stringToNumberMaybe(x) for x in kv[1].split()]
      if len(val_list) != 1:
        d[kv[0]] = val_list
      else:
        d[kv[0]] = val_list[0]
    except IndexError:
      pass
  return d


def readConfigFile(filename):
  """Read a config file and parse it with parseConfig()."""
  fid = open(filename, 'r')
  s = fid.read()
  fid.close()
  return parseConfig(s)

