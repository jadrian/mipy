from mipy.miraw import *

print("""
Running test suite for miraw. Note that part of this test suite involves
manually confirming that the contents of binary files written to disk are
correct; you'll need a hex editor for this.
If you're an end user, you probably didn't mean to run this test suite. If
you just wanted to use the miraw tools, you should just import miraw directly:
  from mipy import miraw
"""[1:])
import struct, sys, binascii
from pprint import pprint as pp

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
true_dimorder = "IXZY"
true_extents = (5,2,4,3)
raw = np.ndarray(true_extents, dtype=np.float32, order='C')
for i in range(raw.shape[0]):
  for z in range(raw.shape[2]):
    for y in range(raw.shape[3]):
      for x in range(raw.shape[1]):
        # Store hex value 'BxByBzBi' for each voxel's x, y, z, i coords.
        raw[i,x,z,y] = hexStrToFloat('b'+'b'.join([str(j) for j in [x,y,z,i]]))

def flatHexList(X):
  return [floatToHexStr(p) for p in X.flat]

def printLongList(L, c=8):
  i = 0
  for p in L:
    sys.stdout.write(" " + p)
    i += 1
    if i % c == 0:
      sys.stdout.write("\n")
  if i % c != 0:
    sys.stdout.write("\n")

raw_hexdump = flatHexList(raw)
# Since raw is in C-style order (last index is fastest-changing), the
# Y index changes fastest, then Z, then X, then I.  So the flat order is:
expected_hexdump = [
  "b0b0b0b0", "b0b1b0b0", "b0b2b0b0", "b0b0b1b0", "b0b1b1b0", "b0b2b1b0", "b0b0b2b0", "b0b1b2b0", "b0b2b2b0", "b0b0b3b0", "b0b1b3b0", "b0b2b3b0",
  "b1b0b0b0", "b1b1b0b0", "b1b2b0b0", "b1b0b1b0", "b1b1b1b0", "b1b2b1b0", "b1b0b2b0", "b1b1b2b0", "b1b2b2b0", "b1b0b3b0", "b1b1b3b0", "b1b2b3b0",
  "b0b0b0b1", "b0b1b0b1", "b0b2b0b1", "b0b0b1b1", "b0b1b1b1", "b0b2b1b1", "b0b0b2b1", "b0b1b2b1", "b0b2b2b1", "b0b0b3b1", "b0b1b3b1", "b0b2b3b1",
  "b1b0b0b1", "b1b1b0b1", "b1b2b0b1", "b1b0b1b1", "b1b1b1b1", "b1b2b1b1", "b1b0b2b1", "b1b1b2b1", "b1b2b2b1", "b1b0b3b1", "b1b1b3b1", "b1b2b3b1",
  "b0b0b0b2", "b0b1b0b2", "b0b2b0b2", "b0b0b1b2", "b0b1b1b2", "b0b2b1b2", "b0b0b2b2", "b0b1b2b2", "b0b2b2b2", "b0b0b3b2", "b0b1b3b2", "b0b2b3b2",
  "b1b0b0b2", "b1b1b0b2", "b1b2b0b2", "b1b0b1b2", "b1b1b1b2", "b1b2b1b2", "b1b0b2b2", "b1b1b2b2", "b1b2b2b2", "b1b0b3b2", "b1b1b3b2", "b1b2b3b2",
  "b0b0b0b3", "b0b1b0b3", "b0b2b0b3", "b0b0b1b3", "b0b1b1b3", "b0b2b1b3", "b0b0b2b3", "b0b1b2b3", "b0b2b2b3", "b0b0b3b3", "b0b1b3b3", "b0b2b3b3",
  "b1b0b0b3", "b1b1b0b3", "b1b2b0b3", "b1b0b1b3", "b1b1b1b3", "b1b2b1b3", "b1b0b2b3", "b1b1b2b3", "b1b2b2b3", "b1b0b3b3", "b1b1b3b3", "b1b2b3b3",
  "b0b0b0b4", "b0b1b0b4", "b0b2b0b4", "b0b0b1b4", "b0b1b1b4", "b0b2b1b4", "b0b0b2b4", "b0b1b2b4", "b0b2b2b4", "b0b0b3b4", "b0b1b3b4", "b0b2b3b4",
  "b1b0b0b4", "b1b1b0b4", "b1b2b0b4", "b1b0b1b4", "b1b1b1b4", "b1b2b1b4", "b1b0b2b4", "b1b1b2b4", "b1b2b2b4", "b1b0b3b4", "b1b1b3b4", "b1b2b3b4"
]
if raw_hexdump == expected_hexdump:
  print("[ PASS ]  Raw memory layout is as expected.")
else:
  print("[ FAIL ]  Raw memory layout does not match expectation:")
  print("          Observed:")
  printLongList(raw_hexdump)
  print("          Expected:")
  printLongList(expected_hexdump)
print()

raw_dest = '/tmp/miraw_test_raw'
size_info_dest = '/tmp/size_info'

saveRaw(raw_dest, raw, order='F', dtype_as_ext=True)
print("[ CHECK ] Just wrote the raw file to %s.float32." % raw_dest)
print("""          Open this file in a hex editor and confirm that it has
          the following 32-bit hex values in order.   Note, though,
          that these values are displayed here in big-endian order;
          if you're on a little-endian architecture (which is very
          likely), then the bytes in each 32-bit word will be stored
          in reverse order.
            b0b0b0b0 b0b0b0b1 b0b0b0b2 b0b0b0b3 b0b0b0b4
            b1b0b0b0 b1b0b0b1 b1b0b0b2 b1b0b0b3 b1b0b0b4
            
            b0b0b1b0 b0b0b1b1 b0b0b1b2 b0b0b1b3 b0b0b1b4
            b1b0b1b0 b1b0b1b1 b1b0b1b2 b1b0b1b3 b1b0b1b4
            
            b0b0b2b0 b0b0b2b1 b0b0b2b2 b0b0b2b3 b0b0b2b4
            b1b0b2b0 b1b0b2b1 b1b0b2b2 b1b0b2b3 b1b0b2b4
            
            b0b0b3b0 b0b0b3b1 b0b0b3b2 b0b0b3b3 b0b0b3b4
            b1b0b3b0 b1b0b3b1 b1b0b3b2 b1b0b3b3 b1b0b3b4
            
            b0b1b0b0 b0b1b0b1 b0b1b0b2 b0b1b0b3 b0b1b0b4
            b1b1b0b0 b1b1b0b1 b1b1b0b2 b1b1b0b3 b1b1b0b4
            
            b0b1b1b0 b0b1b1b1 b0b1b1b2 b0b1b1b3 b0b1b1b4
            b1b1b1b0 b1b1b1b1 b1b1b1b2 b1b1b1b3 b1b1b1b4
            
            b0b1b2b0 b0b1b2b1 b0b1b2b2 b0b1b2b3 b0b1b2b4
            b1b1b2b0 b1b1b2b1 b1b1b2b2 b1b1b2b3 b1b1b2b4
            
            b0b1b3b0 b0b1b3b1 b0b1b3b2 b0b1b3b3 b0b1b3b4
            b1b1b3b0 b1b1b3b1 b1b1b3b2 b1b1b3b3 b1b1b3b4
            
            b0b2b0b0 b0b2b0b1 b0b2b0b2 b0b2b0b3 b0b2b0b4
            b1b2b0b0 b1b2b0b1 b1b2b0b2 b1b2b0b3 b1b2b0b4
            
            b0b2b1b0 b0b2b1b1 b0b2b1b2 b0b2b1b3 b0b2b1b4
            b1b2b1b0 b1b2b1b1 b1b2b1b2 b1b2b1b3 b1b2b1b4
            
            b0b2b2b0 b0b2b2b1 b0b2b2b2 b0b2b2b3 b0b2b2b4
            b1b2b2b0 b1b2b2b1 b1b2b2b2 b1b2b2b3 b1b2b2b4
            
            b0b2b3b0 b0b2b3b1 b0b2b3b2 b0b2b3b3 b0b2b3b4
            b1b2b3b0 b1b2b3b1 b1b2b3b2 b1b2b3b3 b1b2b3b4
""")


saveSizeInfo(size_info_dest, raw, vox_sz=(1,1,1), dimorder=true_dimorder,
              size_cfg={}, infer_name=False)
print("[ CHECK ] Just wrote the size info file to %s." % size_info_dest)
print("""          Open this file in a plaintext editor and confirm that it
          has an image size of (2, 3, 4), num_dwis = 5, and
          dimension order IXZY.""")


(raw_rec, cfg) = readRawWithSizeInfo(raw_dest + ".float32",
                    sizefile=None, dtype=None, cropped=None, dimorder=None,
                    diskorder='F', memorder='C')
# raw_rec and raw should be identical.
if np.array_equal(raw, raw_rec):
  print("[ PASS ]  Successfully wrote a raw image to disk and re-read it.")
else:
  print("[ FAIL ]  Image->disk->image translation error.  Dump:")
  print('          Original image (raw):')
  printLongList(flatHexList(raw))
  print('          Image read from disk (raw_rec):')
  printLongList(flatHexList(raw_rec))
print()

# Write to a single 4-D NIfTI file, using all of our nifty inference
# abilities: we infer the size_info filename from the raw filename, and we
# read the dimension order from the size_info file.
rawToNifti(raw_dest + ".float32", sizefile=None, outfile=None,
            dimorder=None, diskorder='F', dtype=None, split4=False)
print("[ CHECK ] Just wrote the image as a 4-D NIfTI file: %s.nii." % raw_dest)
print("""          Use nifti_tool to check that the "dim" value in the header
          is [4,2,3,4,5,1,1,1] (a 4-D 2x3x4x5 volume).  Then open it in a hex
          editor and look at the data portion, which starts after the end of 
          the header.  (You can spot the end of the header by the magic string
          "n+1", with five null bytes after it.)
            Confirm that the most significant byte of each value is fastest-
          changing, followed by the next-most-significant, and so on.  In
          big-endian format, the expected values are:
            b0b0b0b0 b1b0b0b0 b0b1b0b0 b1b1b0b0 b0b2b0b0 b1b2b0b0
            b0b0b1b0 b1b0b1b0 b0b1b1b0 b1b1b1b0 b0b2b1b0 b1b2b1b0
            b0b0b2b0 b1b0b2b0 b0b1b2b0 b1b1b2b0 b0b2b2b0 b1b2b2b0
            b0b0b3b0 b1b0b3b0 b0b1b3b0 b1b1b3b0 b0b2b3b0 b1b2b3b0
            
            b0b0b0b1 b1b0b0b1 b0b1b0b1 b1b1b0b1 b0b2b0b1 b1b2b0b1
            b0b0b1b1 b1b0b1b1 b0b1b1b1 b1b1b1b1 b0b2b1b1 b1b2b1b1
            b0b0b2b1 b1b0b2b1 b0b1b2b1 b1b1b2b1 b0b2b2b1 b1b2b2b1
            b0b0b3b1 b1b0b3b1 b0b1b3b1 b1b1b3b1 b0b2b3b1 b1b2b3b1
            
            b0b0b0b2 b1b0b0b2 b0b1b0b2 b1b1b0b2 b0b2b0b2 b1b2b0b2
            b0b0b1b2 b1b0b1b2 b0b1b1b2 b1b1b1b2 b0b2b1b2 b1b2b1b2
            b0b0b2b2 b1b0b2b2 b0b1b2b2 b1b1b2b2 b0b2b2b2 b1b2b2b2
            b0b0b3b2 b1b0b3b2 b0b1b3b2 b1b1b3b2 b0b2b3b2 b1b2b3b2
            
            b0b0b0b3 b1b0b0b3 b0b1b0b3 b1b1b0b3 b0b2b0b3 b1b2b0b3
            b0b0b1b3 b1b0b1b3 b0b1b1b3 b1b1b1b3 b0b2b1b3 b1b2b1b3
            b0b0b2b3 b1b0b2b3 b0b1b2b3 b1b1b2b3 b0b2b2b3 b1b2b2b3
            b0b0b3b3 b1b0b3b3 b0b1b3b3 b1b1b3b3 b0b2b3b3 b1b2b3b3
            
            b0b0b0b4 b1b0b0b4 b0b1b0b4 b1b1b0b4 b0b2b0b4 b1b2b0b4
            b0b0b1b4 b1b0b1b4 b0b1b1b4 b1b1b1b4 b0b2b1b4 b1b2b1b4
            b0b0b2b4 b1b0b2b4 b0b1b2b4 b1b1b2b4 b0b2b2b4 b1b2b2b4
            b0b0b3b4 b1b0b3b4 b0b1b3b4 b1b1b3b4 b0b2b3b4 b1b2b3b4
""")

# Same as above, but with split files.
rawToNifti('/tmp/miraw_test_raw.float32', sizefile=None, outfile=None,
            dimorder=None, diskorder='F', dtype=None, split4=True)
print("[ CHECK ] Just wrote the image as 5 3-D NIfTI files: %s_XXX.nii." % raw_dest)
print("""          There should be five volumes, numbered 000 through 004, one
          per "I" value.  Use nifti_tool to check that the "dim" value in the
          header of each file is [3,2,3,4,1,1,1,1] (a 3-D 2x3x4 volume).  Then
          open each one in a hex editor.  You should see the values from each
          respective group above in the data portion of the file.
""")

