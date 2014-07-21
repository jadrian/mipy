import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

"""Viewing functions for raw medical image data.

Before calling any of these functions, be sure to have run miview.plt.ion();
this turns on interactive plotting, which is necessary for the first function
call to successfully create a figure window.  Otherwise, images will be
"displayed" in a virtual window that you won't see, and won't necessarily have
easy programmatic access to."""


greymap_with_gamma = {'red':   [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
                      'green': [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
                      'blue':  [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]}

# Feature requests:
# - Enable flagging of out-of-bounds values

def viewSlice(X, lo=None, hi=None, gamma=0.5, flipaxes=False):
  """View a 2-D image: either scalars mapped to greyscale, or RGB[A].
  
  Using matplotlib.pyplot.imshow(), display a properly-oriented image of an
  ndarray: either in greyscale if 2-D, or color if 3-D.
  
  This function assumes that a matplotlib figure is already open.  If not,
  you can force a new one by calling
    miview.plt.ion()
  before your first call to viewSlice().  You may (and should) then call
    miview.plt.ioff()
  
  Arguments:
    X:        An ndarray with shape (A,B), (A,B,3), or (A,B,4).
    lo:       The value to map to pure black (automatically computed if None).
    hi:       The value to map to pure white (automatically computed if None).
    gamma:    The gamma (mid-tone warping) to apply to the greyscale map.
    flipaxes: If False (default), map the first dimension of X to the horizontal
              plot axis.  If True, map to the vertical instead.
  
  Examples:
  # Here foo is an ndarry with foo.shape = (64, 36, 28, 22).
  # Note that foo[:,5,:,18].shape = (64, 28): it's a 2-D subvolume.
  
  viewSlice(foo[:,5,:,18])
  # The result is that the given slice of foo is displayed in a pyplot figure,
  # where dimension 0 of foo is mapped to the horizontal axis (increasing left
  # to right), and dimension 2 of foo is mapped to the vertical axis (increasing
  # bottom to top).
  
  # Luminosity is normalized within the slice.  To normalize across the
  # whole volume:
  viewSlice(foo[:,5,:,18], lo=min(foo.flat), hi=max(foo.flat))
  
  # Here bar is an ndarry with bar.shape = (64, 36, 28, 22, 3).
  # Note that bar[:,5,:,18,:].shape = (64, 28, 3): it's a 3-D subvolume, with
  # last dimension of extent 3, which can therefore be interpreted as RGB.
  
  viewSlice(bar[:,5,:,18,:])
  # The result is that the given slice of bar is displayed in a pyplot figure,
  # where dimension 0 of foo is mapped to the horizontal axis, increasing from
  # left to right, dimension 2 of foo is mapped to the vertical axis, increasing
  # from bottom to top, and dimension 4 is interpreted to contain R, G, and B
  # values that are automatically rescaled to be between 0 and 1.
  
  # To manually override the luminance rescaling, provide values for lo and hi.
  # Specifically, if you already have a properly-scaled RGB image with values
  # between 0 and 1, but not necessarily containing pure black, white, red,
  # etc., then you want:
  viewSlice(bar[:,5,:,18,:], lo=0, hi=1)
  """
  scalar = (len(X.shape) == 2)
  rgb = (len(X.shape) == 3 and (X.shape[2] == 3 or X.shape[2] == 4))
  if not (scalar or rgb):
    raise IndexError('X must be 2-D or 3-D with X.shape[2] = 3 or 4.')
  scale = colors.Normalize(lo, hi, clip=True)
  scale.autoscale_None(X)
  # Colormap is ignored if X is an RGB[A] image.
  gmap = colors.LinearSegmentedColormap('anon', greymap_with_gamma, gamma=gamma)
  tr = [1,0]
  if flipaxes:
    tr = [0,1]
  if rgb:
    tr += [2]
  plt.imshow(X.transpose(tr), norm=scale, cmap=gmap,
             origin='lower', interpolation='nearest')
  plt.draw()


def stackToVideo(X, t=0.1, lo=None, hi=None,
                 dims=(0,1,2), gamma=0.5, flipaxes=False):
  """View a 3-D image volume as an animation: either in greyscale or in color.
  
  This is like calling viewSlice() a bunch of times in succession, with a
  time.sleep(t) in between, but way more efficient.
  
  This function assumes that a matplotlib figure is already open.  If not,
  you can force a new one by calling
    miview.plt.ion()
  before your first call to stackToVideo().  You may (and should) then call
    miview.plt.ioff()
  
  You can stop the video at any time by hitting ctrl-C.
  
  Arguments:
    X:        An ndarray with shape (A,B,C), (A,B,C,3), or (A,B,C,4).
    t:        The pause time between frames, in seconds.
    lo:       The value to map to pure black (automatically computed if None).
    hi:       The value to map to pure white (automatically computed if None).
    dims:     A 3-tuple, default (0,1,2).  The dimensions of X to map to the
              three axes of the display: dims[0] maps to the horizontal axis,
              dims[1] to the vertical, and dims[2] increments through time.
              Note that it is not possible to choose another dimension for color
              data; if X is 4-D, the last dimension will always be used.
    gamma:    The gamma (mid-tone warping) to apply to the greyscale map.
    flipaxes: If False (default), map the first dimension of X to the horizontal
              plot axis.  If True, map to the vertical instead.
  
  Examples:
  # Here foo is an ndarry with foo.shape = (64, 36, 28, 22).  We will call the
  # four dimensions of this ndarray A, B, C, and D respectively.  Note that the
  # 3-D subvolume we use in the examples below, foo[:,5,:,:], is in fact a 3-D
  # array with shape (64, 28, 22) and dimensions A, C, and D.
  
  stackToVideo(foo[:,5,:,:])
  # The result is that a series of 22 animation frames are displayed, each one
  # being equivalent to viewSlice(foo[:,5,:,t]), for t in range(22).
  # The A dimension is mapped to the horizontal axis.
  # The C dimension is mapped to the vertical axis.
  # The D dimension is mapped to the time axis.
  
  # To animate slices along a different dimension, use the "dims" argument:
  stackToVideo(foo[:,5,:,:], dims=(1,2,0))
  # The result is an animation where:
  # The C dimension is mapped to the horizontal axis.
  # The D dimension is mapped to the vertical axis.
  # The A dimension is mapped to the time axis.
  
  # Here bar is an ndarry with bar.shape = (64, 36, 28, 22, 3), and dimension
  # names A, B, C, D, and E, respectively.
  # Note that bar[:,5,:,:,:].shape = (64, 28, 22, 3): it's a 4-D subvolume, with
  # last dimension of extent 3, which can therefore be interpreted as RGB.
  
  stackToVideo(bar[:,5,:,:,:])
  # The result is an animation where:
  # The A dimension is mapped to the horizontal axis.
  # The C dimension is mapped to the vertical axis.
  # The D dimension is mapped to the time axis.
  # The E dimension contains the RGB components.
  
  # See documentation on viewSlice for more details about colors and auto-
  # scaling of the displayed values.
  """
  scalar = (len(X.shape) == 3)
  rgb = (len(X.shape) == 4 and (X.shape[3] == 3 or X.shape[3] == 4))
  if not (scalar or rgb):
    raise IndexError('X must be 3-D or 4-D with X.shape[3] = 3 or 4.')
  if flipaxes:
    dims = (dims[1], dims[0], dims[2])
  if scalar:
    XT = X.transpose(dims[1], dims[0], dims[2])
  else:
    XT = X.transpose(dims[1], dims[0], dims[2], 3)
  scale = colors.Normalize(lo, hi, clip=True)
  scale.autoscale_None(X)
  ax = plt.gca()
  if scalar:
    gmap = colors.LinearSegmentedColormap('anon',greymap_with_gamma,gamma=gamma)
    img = ax.imshow(XT[:,:,0],
                    norm=scale,
                    cmap=gmap,
                    origin='lower',
                    interpolation='nearest')
    for z in range(XT.shape[2]):
      try:
        img.set_data(XT[:,:,z])
        ax.set_title(z)
        plt.draw()
        time.sleep(t)
      except KeyboardInterrupt:
        img.set_data(XT[:,:,z])
        plt.draw()
        break
  else:
    img = ax.imshow(XT[:,:,0],
                    norm=scale,
                    origin='lower',
                    interpolation='nearest')
    for z in range(XT.shape[2]):
      try:
        img.set_data(XT[:,:,z,:])
        plt.draw()
        time.sleep(t)
      except KeyboardInterrupt:
        img.set_data(XT[:,:,z,:])
        plt.draw()
        break
