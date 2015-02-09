import numpy as np
import os.path
import re

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


def detectShapeAndCropping(num_vox, num_uncropped, num_cropped, num_imgs,
                           cropped = None, threeD = None):
  """Detect whether a raw image is cropped and/or 4-D.
  
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
    raise ValueError('Could not infer dtype from filename %s.' % getFilename(f))
  return dtype


# The following two versions of ndcopy are used to work around the bug in numpy
# < 1.8.0 in which copy() doesn't take an order argument.  The workaround code
# is in miraw.py and runs at import time.
def ndcopyUsesOrder(X, order):
  return X.copy(order=order)

def ndcopyIgnoreOrder(X, order):
  return X.copy()


############
# The following is a runtime bugfix for older versions of numpy that don't
# support an "order" argument to either of the numpy "copy()" functions: neither
# to the plain numpy.copy(), nor to the copy() member function of ndarray
# objects.  This bug is confirmed in numpy 1.5.1 (on Python 2.7) and is fixed
# in numpy 1.8.0 (on Python 2.7 and 3.3).

def getBestSupportedNdcopy():
  try:
    import numpy.random
    X = numpy.random.random((5,3,4))
    Y = X.copy(order="F")
    _ndcopyWithOrder = ndcopyUsesOrder
  except TypeError:
    import warnings
    w = "\n  Your current numpy version is " + str(numpy.__version__) + "." + """
    This numpy version does not support an 'order'
  argument to ndarry.copy().  Please update numpy
  to a version >= 1.8.0.
    miraw.readRaw() and miraw.readRawWithSizeInfo()
  will not be able to use custom memory orders; the
  'memorder' keyword argument will be silently
  ignored.
  """
    warnings.warn(w, RuntimeWarning, stacklevel=2)
    _ndcopyWithOrder = ndcopyIgnoreOrder
  return _ndcopyWithOrder

ndcopyWithOrder = getBestSupportedNdcopy()

# End fix for stupid numpy bug.
############
