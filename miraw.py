import numpy as np
import os.path
import re
import time
from nibabel import nifti1

"""Functions for dealing with raw medical imaging datasets.

This module is particularly focused on working with diffusion-weighted images
and derived images, which are typically 4-D (3 for space, plus one dimension for
arbitrary sample vectors).  Its default metadata format, "size_info", is a hacky
thing custom-built just for DWIs, and does not accommodate N-D volumes for N < 3
or N > 4.  The loadRaw() and saveRaw() functions, however, will work for any N.
"""

def loadRaw(f, shape, dtype=None, diskorder='F', memorder='C'):
  """Load array data from a raw binary file on disk.
  
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
    dtype = inferDtypeFromFilename(f)
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
  return ndcopyWithOrder(X, memorder)


def saveRaw(f, X, order='F', dtype_as_ext=False):
  """Save array data to a raw binary file on disk.
  
  This is a wrapper around numpy.ndarray.tofile.  Its particular purpose is
  to enable the creation of "Fortran-ordered" raw files, (aka column-major;
  Matlab's default), in which the fastest-changing index in the source array,
  with respect to the linear order in which data are stored on disk, is the
  first index, rather than the last index ("C-ordered", numpy's default).
  
  Arguments:
    f:     An open file object or a filename.
    X:     A numpy ndarray, with any shape and storage order.
    order: 'F' or 'C' --- the order for storage on disk.
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

DIM_DEFAULT_ORDER = 'XYZI'
DIM_INDICES = {'X':0, 'Y':1, 'Z':2, 'I':3}

def isValidDimorder(s):
  try:
    return (''.join(sorted(s)) == 'IXYZ')
  except:
    return False

def dimorderToDimmap(s):
  return [DIM_INDICES[d] for d in s]

def dimorderToReverseDimmap(s):
  return [s.index(d) for d in DIM_DEFAULT_ORDER]

def loadRawWithSizeInfo(f, sizefile=None, dtype=None, cropped=None,
                        dimorder=None, diskorder='F', memorder='C'):
  """Load a raw image file from disk, using a size_info metadata file.
  
  Arguments:
    f:         A filename or open file object for the raw data file.
    sizefile:  A filename for the size_info metadata file.
               If None, look for a file called "size_info" in f's directory.
    dtype:     The numpy dtype of the raw data, or None.
    cropped:   A boolean: whether to use the cropped or full volume, or None.
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
  
  (See loadRaw for more explanation of the last two arguments.)
  
  This function attempts, usually successfully, to infer the values of
  arguments left None.
  
  Returns (vol, cfg), where vol is a numpty ndarray and cfg is the dict of
  settings in sizefile.  In addition, this function defines an additional
  key, cfg['cropped'], with Boolean value.
  """
  
  # Read the image into a 1-D array.
  raw = loadRaw(f, (-1,1), dtype=dtype, diskorder=diskorder, memorder=memorder)
  
  # Read the size file.
  imgname = getFilename(f)
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
  cropped, threeD = detectShapeAndCropping(raw.size,
    np.prod(sz), np.prod(sz_c), n_imgs, cropped)
  if cropped:
    sz = sz_c
  sz = sz + [n_imgs]
  cfg['cropped'] = cropped
  
  # Finally set the size and return.
  try:
    dimorder = cfg['dimension_order'][0]
  except KeyError:
    if dimorder is None:
      dimorder = DIM_DEFAULT_ORDER
  if not isValidDimorder(dimorder):
    raise ValueError('"%s" is not a valid dimorder argument.' % repr(dimorder))
  
  sz = np.take(sz, dimorderToDimmap(dimorder), axis=0)
  return (ndcopyWithOrder(raw.reshape(sz, order=diskorder), memorder), cfg)


def saveSizeInfo(f, img, vox_sz=(1,1,1), dimorder=None, size_cfg={},
                 infer_name=True):
  """Write a size_info metadata file to disk for a given array.
  
  A size_info file stores image (array) dimensions for raw images, as well as
  voxel size and cropping information (indicating that the array is cropped
  from a larger volume).  Note that size_info is designed for 3-D or 4-D
  arrays only, and stores the extents of the first three dimensions
  separately from that of the fourth.
  
  Arguments:
    f:          An open file handle or a filename.
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
                the size of X.  Be careful!  This includes the
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
    dimorder = size_cfg['dimension_order'][0]
  except KeyError:
    if dimorder is None:
      dimorder = DIM_DEFAULT_ORDER
  if not isValidDimorder(dimorder):
    raise ValueError('"%s" is not a valid dimorder argument.' % repr(dimorder))
  shape = np.take(shape, dimorderToReverseDimmap(dimorder), axis=0).tolist()
  
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
    auto_cfg[k] = a
  
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
  
  lines = cleanlines(string.splitlines())
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


def cleanlines(lines):
  """Remove comments and blank lines from splitlines output."""
  # Clean comments.
  matchRE = re.compile('(.*?)(//|%|#)')
  for i in range(len(lines)):
    line = lines[i]
    match = matchRE.match(line)
    if match is not None:
      lines[i] = match.group(1)
  # Clean blank lines.
  return [x.strip() for x in lines if len(x.strip()) > 0]


def rawToNifti(infile, sizefile=None, outfile=None, dimorder=None,
               mapper=None, diskorder='F', dtype=None, split4=False):
  """Convert a raw file to a NIfTI file.
  
  Arguments:
    infile:    filename of a raw file.
    sizefile:  filename of a size_info config file.
    outfile:   filename (including .nii) of the NIfTI file to generate.
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
    mapper:    a function to convert voxel values in place.
    diskorder: A string, 'F' or 'C', representing the order in which the data
               values are stored in the raw file.
    dtype:     the numpy dtype for the infile.  If None, it is inferred from
               infile's extension.
    split4:    If True, output numbered 3-D images from 4-D input.
  """
  (raw, cfg) = loadRawWithSizeInfo(infile, sizefile=sizefile, dtype=dtype,
                                   dimorder=dimorder, diskorder=diskorder,
                                   memorder='C')
  vox_sz = cfg['voxel_size_(mm)']
  
  # Rearrange dimensions.
  try:
    dimorder = cfg['dimension_order'][0]
  except KeyError:
    if dimorder is None:
      dimorder = DIM_DEFAULT_ORDER
  if not isValidDimorder(dimorder):
    raise ValueError('"%s" is not a valid dimorder argument.' % repr(dimorder))
  raw_transp = raw.transpose(dimorderToReverseDimmap(dimorder))
  
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


def detectShapeAndCropping(num_vox, num_uncropped, num_cropped, num_imgs,
                           cropped = None, threeD = None):
  """Detect whether a Blockhead raw image is cropped and/or 4-D.
  
  Detects and/or checks the values of its last two arguments.  If a non-None
  value is given for either one, but that information is inconsistent with
  the different image sizes, this function raises a ValueError.
  
  Arguments:
    num_vox:       The number of voxels in the raw image in question.
    num_uncropped: The number of voxels a 3-D uncropped image would have.
    num_cropped:   The number of voxels a 3-D cropped image would have.
    num_imgs:      The number of images/acquisitions a 4-D volume would have.
    cropped:       Boolean or None: whether this image is cropped.
    threeD:        Boolean or None: whether this image is 3-D (False=4-D).
  
  Returns (cropped, threeD): the consistent values of those aguments.
  """
  if cropped is not None:
    if threeD is not None:
      expected_num = 1
      if cropped:
        expected_num = num_cropped
      else:
        expected_num = num_uncropped
      if not threeD:
        expected_num *= num_imgs
      if expected_num != num_vox:
        raise ValueError('Mismatch between claimed and observed sizes.')
    else:
      expected_num = 1
      if cropped:
        expected_num = num_cropped
      else:
        expected_num = num_uncropped
      num_imgs_needed = int(num_vox / expected_num)
      if num_imgs_needed * expected_num != num_vox:
        raise ValueError('Observed size not achievable with claimed cropping.')
      if num_imgs_needed == 1:
        threeD = True
      elif num_imgs_needed == num_imgs:
        threeD = False
      else:
        raise ValueError("Observed size doesn't match num_imgs.")
  else:
    try:
      cropped, threeD = detectShapeAndCropping(
        num_vox, num_uncropped, num_cropped, num_imgs, True, threeD)
    except:
      try:
        cropped, threeD = detectShapeAndCropping(
          num_vox, num_uncropped, num_cropped, num_imgs, False, threeD)
      except:
        raise ValueError('No good interpretation of the observed size.')
  return cropped, threeD


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
    s: a string containing the full contents of a config file.
  
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


def getFilename(f):
  """Get the filename from an open file handle or filename string."""
  if isinstance(f, str):
    return f
  return f.name


def inferDtypeFromFilename(f):
  """Infer the numpy dtype of a raw file from its filename extension."""
  try:
    dtype = np.dtype(os.path.splitext(getFilename(f))[1][1:])
  except:
    raise ValueError('Could not infer dtype from filename %s.' % name)
  return dtype


def test():
  import struct, sys, binascii
  
  def hexStrToFloat_str_decode(s):
    return struct.unpack('!f', s.decode('hex'))[0]
  
  def floatToHexStr_str_encode(v):
    return struct.pack('!f', v).encode('hex')
  
  def hexStrToFloat_bytes_fromhex(s):
    return struct.unpack('!f', bytes.fromhex(s))[0]
  
  def floatToHexStr_binascii(v):
    return str(binascii.hexlify(struct.pack('!f', v)), 'ascii')
  
  if hasattr(str, 'decode') and hasattr(str, 'decode'):
    # Python 2.x: str.decode.
    hexStrToFloat = hexStrToFloat_str_decode
    floatToHexStr = floatToHexStr_str_encode
  elif hasattr(bytes, 'fromhex'):
    # Python 3.x: bytes.fromhex.
    hexStrToFloat = hexStrToFloat_bytes_fromhex
    floatToHexStr = floatToHexStr_binascii
  else:
    raise NameError('No known valid hex-to-float conversion method.')
  
  # Our raw file's dimension order is (I,X,Z,Y), just to be obnoxious.
  # Extents are (X,Y,Z,I) = (2,3,4,5).
  raw = np.ndarray((5,2,4,3), dtype=np.float32, order='C')
  for i in range(raw.shape[0]):
    for z in range(raw.shape[2]):
      for y in range(raw.shape[3]):
        for x in range(raw.shape[1]):
          # Store hex value 'BxByBzBi' for each voxel's x, y, z, i coords.
          raw[i,x,z,y] = hexStrToFloat('b'+'b'.join([str(j) for j in [x,y,z,i]]))
  
  def printHexFlat(X, m=8):
    i = 0
    for p in X.flat:
      sys.stdout.write(' ' + floatToHexStr(p))
      i += 1
      if i % m == 0:
        sys.stdout.write('\n')
    if i % m != 0:
      sys.stdout.write('\n')
  
  printHexFlat(raw, 12)
  # Since raw is in C-style order (last index is fastest-changing), the
  # Y index changes fastest, then Z, then X, then I.  So the flat order is:
  #  b0b0b0b0 b0b1b0b0 b0b2b0b0 b0b0b1b0 b0b1b1b0 b0b2b1b0 b0b0b2b0 b0b1b2b0 b0b2b2b0 b0b0b3b0 b0b1b3b0 b0b2b3b0
  #  b1b0b0b0 b1b1b0b0 b1b2b0b0 b1b0b1b0 b1b1b1b0 b1b2b1b0 b1b0b2b0 b1b1b2b0 b1b2b2b0 b1b0b3b0 b1b1b3b0 b1b2b3b0
  #  b0b0b0b1 b0b1b0b1 b0b2b0b1 b0b0b1b1 b0b1b1b1 b0b2b1b1 b0b0b2b1 b0b1b2b1 b0b2b2b1 b0b0b3b1 b0b1b3b1 b0b2b3b1
  #  b1b0b0b1 b1b1b0b1 b1b2b0b1 b1b0b1b1 b1b1b1b1 b1b2b1b1 b1b0b2b1 b1b1b2b1 b1b2b2b1 b1b0b3b1 b1b1b3b1 b1b2b3b1
  #  b0b0b0b2 b0b1b0b2 b0b2b0b2 b0b0b1b2 b0b1b1b2 b0b2b1b2 b0b0b2b2 b0b1b2b2 b0b2b2b2 b0b0b3b2 b0b1b3b2 b0b2b3b2
  #  b1b0b0b2 b1b1b0b2 b1b2b0b2 b1b0b1b2 b1b1b1b2 b1b2b1b2 b1b0b2b2 b1b1b2b2 b1b2b2b2 b1b0b3b2 b1b1b3b2 b1b2b3b2
  #  b0b0b0b3 b0b1b0b3 b0b2b0b3 b0b0b1b3 b0b1b1b3 b0b2b1b3 b0b0b2b3 b0b1b2b3 b0b2b2b3 b0b0b3b3 b0b1b3b3 b0b2b3b3
  #  b1b0b0b3 b1b1b0b3 b1b2b0b3 b1b0b1b3 b1b1b1b3 b1b2b1b3 b1b0b2b3 b1b1b2b3 b1b2b2b3 b1b0b3b3 b1b1b3b3 b1b2b3b3
  #  b0b0b0b4 b0b1b0b4 b0b2b0b4 b0b0b1b4 b0b1b1b4 b0b2b1b4 b0b0b2b4 b0b1b2b4 b0b2b2b4 b0b0b3b4 b0b1b3b4 b0b2b3b4
  #  b1b0b0b4 b1b1b0b4 b1b2b0b4 b1b0b1b4 b1b1b1b4 b1b2b1b4 b1b0b2b4 b1b1b2b4 b1b2b2b4 b1b0b3b4 b1b1b3b4 b1b2b3b4
  
  saveRaw('/tmp/miraw_test_raw', raw, order='F', dtype_as_ext=True)
  # The raw file should have these (32-bit) hex values in order (though
  # note that machine endianness might flip them around!):
  # b0b0b0b0, b0b0b0b1, b0b0b0b2, b0b0b0b3, b0b0b0b4,
  # b1b0b0b0, b1b0b0b1, b1b0b0b2, b1b0b0b3, b1b0b0b4,
  # b0b0b1b0, b0b0b1b1, b0b0b1b2, b0b0b1b3, b0b0b1b4,
  # b1b0b1b0, b1b0b1b1, b1b0b1b2, b1b0b1b3, b1b0b1b4,
  # b0b0b2b0, b0b0b2b1, b0b0b2b2, b0b0b2b3, b0b0b2b4,
  # b1b0b2b0, b1b0b2b1, b1b0b2b2, b1b0b2b3, b1b0b2b4,
  # b0b0b3b0, b0b0b3b1, b0b0b3b2, b0b0b3b3, b0b0b3b4,
  # b1b0b3b0, b1b0b3b1, b1b0b3b2, b1b0b3b3, b1b0b3b4,
  # b0b1b0b0, b0b1b0b1, b0b1b0b2, b0b1b0b3, b0b1b0b4,
  # b1b1b0b0, b1b1b0b1, b1b1b0b2, b1b1b0b3, b1b1b0b4,
  # b0b1b1b0, b0b1b1b1, b0b1b1b2, b0b1b1b3, b0b1b1b4,
  # b1b1b1b0, b1b1b1b1, b1b1b1b2, b1b1b1b3, b1b1b1b4,
  # b0b1b2b0, b0b1b2b1, b0b1b2b2, b0b1b2b3, b0b1b2b4,
  # b1b1b2b0, b1b1b2b1, b1b1b2b2, b1b1b2b3, b1b1b2b4,
  # b0b1b3b0, b0b1b3b1, b0b1b3b2, b0b1b3b3, b0b1b3b4,
  # b1b1b3b0, b1b1b3b1, b1b1b3b2, b1b1b3b3, b1b1b3b4,
  # b0b2b0b0, b0b2b0b1, b0b2b0b2, b0b2b0b3, b0b2b0b4,
  # b1b2b0b0, b1b2b0b1, b1b2b0b2, b1b2b0b3, b1b2b0b4,
  # b0b2b1b0, b0b2b1b1, b0b2b1b2, b0b2b1b3, b0b2b1b4,
  # b1b2b1b0, b1b2b1b1, b1b2b1b2, b1b2b1b3, b1b2b1b4,
  # b0b2b2b0, b0b2b2b1, b0b2b2b2, b0b2b2b3, b0b2b2b4,
  # b1b2b2b0, b1b2b2b1, b1b2b2b2, b1b2b2b3, b1b2b2b4,
  # b0b2b3b0, b0b2b3b1, b0b2b3b2, b0b2b3b3, b0b2b3b4,
  # b1b2b3b0, b1b2b3b1, b1b2b3b2, b1b2b3b3, b1b2b3b4.
  
  saveSizeInfo('/tmp/size_info', raw, vox_sz=(1,1,1), dimorder='IXZY',
               size_cfg={}, infer_name=False)
  # The size_info file should indicate (X,Y,Z) = (2,3,4), with num_dwis=5.
  
  (raw_rec, cfg) = loadRawWithSizeInfo('/tmp/miraw_test_raw.float32',
                     sizefile=None, dtype=None, cropped=None, dimorder=None,
                     diskorder='F', memorder='C')
  # raw_rec and raw should be identical.
  if np.array_equal(raw, raw_rec):
    print('Successfully wrote a raw image to disk and re-read it.')
  else:
    print('Image->disk->image translation error.  Dump:')
    print('raw:')
    printHexFlat(raw, 12)
    print('raw_rec:')
    printHexFlat(raw_rec, 12)
  
  rawToNifti('/tmp/miraw_test_raw.float32', sizefile=None, outfile=None,
             dimorder=None, mapper=None, diskorder='F', dtype=None, split4=False)
  # The data portion of the NIfTI file should have the hex values increasing
  # leftmost-fastest: b0b0b0b0, b1b0b0b0, b0b1b0b0, b1b1b0b0, b0b2b0b0, ...
  # Note, though, that the machine endianness might flip all these around!
  
  rawToNifti('/tmp/miraw_test_raw.float32', sizefile=None, outfile=None,
             dimorder=None, mapper=None, diskorder='F', dtype=None, split4=True)
  # The data portion of the NIfTI file should have the hex values increasing
  # leftmost-fastest: b0b0b0b0, b1b0b0b0, b0b1b0b0, b1b1b0b0, b0b2b0b0, ...
  # Note, though, that the machine endianness might flip all these around!
  

############
# The following is a runtime bugfix for older versions of numpy that don't
# support an "order" argument to either of the numpy "copy()" functions: neither
# to the plain numpy.copy(), nor to the copy() member function of ndarray
# objects.  This bug is confirmed in numpy 1.5.1 (on Python 2.7) and is fixed
# in numpy 1.8.0 (on Python 2.7 and 3.3).

def ndcopyUsesOrder(X, order):
  return X.copy(order=order)

def ndcopyIgnoreOrder(X, order):
  return X.copy()

try:
  import numpy.random
  X = numpy.random.random((5,3,4))
  Y = X.copy(order="F")
  ndcopyWithOrder = ndcopyUsesOrder
except TypeError:
  import warnings
  w = "\n  Your current numpy version is " + str(numpy.__version__) + "." + """
  This numpy version does not support an 'order'
argument to ndarry.copy().  Please update numpy
to a version >= 1.8.0.
  miraw.loadRaw() and miraw.loadRawWithSizeInfo()
will not be able to use custom memory orders; the
'memorder' keyword argument will be silently
ignored.
"""
  warnings.warn(w, RuntimeWarning, stacklevel=2)
  ndcopyWithOrder = ndcopyIgnoreOrder

# End fix for stupid numpy bug.
############
