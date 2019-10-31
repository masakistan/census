import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from os.path import join
from os import makedirs

START, END = 280, 735
THRESH = 300000
VTHRESH = 60000
HEADER_START_THRESH = 32500
PAD = 0
MERGE_THRESH = 1
VERTICAL_START = 20
VERTICAL_END = 2450

FILTER_THRESH = 3
EXPECTED_HEIGHT = 40
EXPECTED_HEIGHT_TOLERANCE = 10
MERGED_ROW_TOLERANCE = 10
COL_END_THRESH = 100
COL_START_THRESH = 500
TOP_CELL_BUFFER = 2
BOTTOM_CELL_BUFFER = 20

BIN_THRESH = 220


def filter(z, p, v, z_thresh):
    n_z, n_p, n_v = [], [], []
    for _z, _p, _v in zip(z, p, v):
        if _z > z_thresh:
            n_z.append(_z)
            n_p.append(_p)
            n_v.append(_v)
    return np.array(n_z), np.array(n_p), np.array(n_v)


def rle(inarray):
    """ run length encoding. Partial credit to R rle function. 
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values) """
    ia = np.asarray(inarray)                  # force numpy
    n = len(ia)
    if n == 0: 
        return (None, None, None)
    else:
        y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)   # must include last element posi
        z = np.diff(np.append(-1, i))       # run lengths
        p = np.cumsum(np.append(0, z))[:-1] # positions
        return(z, p, ia[i])


def merge(z, p, thresh):
    n_z, n_p = [], []

    s_z, s_p = z[0], p[0]
    for i in range(1, len(z)):
        #print('current:', s_p, s_z)
        if abs(s_p + s_z - p[i]) < thresh:
            s_z += z[i]
        else:
            n_z.append(s_z)
            n_p.append(s_p)
            s_z = z[i]
            s_p = p[i]
            
    n_z.append(s_z)
    n_p.append(s_p)
    return np.array(n_z), np.array(n_p)

    
def split_erroneously_merged_rows(z, p):
    n_z, n_p = [], []
    for _z, _p in zip(z, p):
        #print(_p, _z)
        #if _z > (EXPECTED_HEIGHT * 2) - MERGED_ROW_TOLERANCE and (_z % EXPECTED_HEIGHT < MERGED_ROW_TOLERANCE or _z % EXPECTED_HEIGHT > EXPECTED_HEIGHT - MERGED_ROW_TOLERANCE):
        if _z > EXPECTED_HEIGHT + MERGED_ROW_TOLERANCE:
            #if _z % EXPECTED_HEIGHT < MERGED_ROW_TOLERANCE:
            factor = _z // EXPECTED_HEIGHT
            if _z % EXPECTED_HEIGHT > EXPECTED_HEIGHT - MERGED_ROW_TOLERANCE:
                #factor = (_z // EXPECTED_HEIGHT) + 1
                factor += 1
            #print("\tfound row to be split", _p, _z, 'factor:', factor)
            #print(factor)
            for i in range(factor):
                height = _z // factor
                start = _p + (i * height)
                #print(_p, start, _z, height)
                n_z.append(1)
                n_z.append(height)
                n_z.append(1)
                n_p.append(start - 1)
                n_p.append(start)
                n_p.append(start + height)
        else:
            n_z.append(_z)
            n_p.append(_p)
    return n_z, n_p


img = cv2.imread(sys.argv[1])
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#print(img_gray.shape)
#img_bin = img_gray
#img_bin[img_bin < BIN_THRESH] = 0
#img_bin[img_bin >= BIN_THRESH] = 255


#plt.hist(img_gray.flatten(), bins = 'auto')
#plt.show()

# NOTE: names should be somewhere in here
img_bin = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 11)
img_bin_piece = img_bin[:, START:END]
#cv2.imwrite('ex_binaraized_adaptive.jpg', img_bin_piece)

#img_bin_piece = img_bin[:, START:END]
#cv2.imwrite('ex_binarized.jpg', img_bin_piece)

hdist = np.sum(img_bin_piece, axis = 0)

boundaries = hdist > THRESH
z, p, v = rle(boundaries)
b_z, b_p = None, None
dist = float('inf')
for i in zip(p, z, v):
    #print(dist)
    if i[2]:
        if abs(i[1] - 400) < dist:
            dist = abs(i[1] - 400)
            b_z = i[1]
            b_p = i[0]
    #print(i)

assert b_z > 360, 'ERROR: name column too small at ' + str(b_z)
name = img[:, START + b_p - PAD : START + b_p + b_z + PAD]
name_bin = img_bin[:, START + b_p - PAD : START + b_p + b_z + PAD]
#cv2.imwrite('col_1_before.jpg', name_bin)

vdist = np.sum(name_bin, axis = 1)
#plt.plot([i for i in range(len(vdist))], vdist)
#plt.show()

col_boundaries = vdist < HEADER_START_THRESH
cz, cp, cv = rle(col_boundaries)
#print(cp, cv)
cp = cp[cv]
#print(cp)
if len(cp) == 0:
    col_start = 0
    col_end = len(name_bin)
else:
    col_start = cp[0]
    col_end = cp[-1]

    if col_start > COL_START_THRESH:
        col_start = 200

    if col_end < len(name_bin) - COL_END_THRESH:
        col_end = len(name_bin)
#print(header_start)
name_bin = name_bin[col_start:col_end]
#cv2.imwrite('col_2_fixed.jpg', name_bin)
vdist = np.sum(name_bin, axis = 1)

#vdist = vdist[:1000] / 100000


def test_func(x, a, b):
    return a * np.cos(b * x)


#x_data = np.array([i for i in range(len(vdist))])
#params, params_covariance = optimize.curve_fit(test_func, x_data, vdist, p0 = [2, 2])
#print(params)
out_dir = sys.argv[2]
#print('out dir', out_dir)
try:
    makedirs(out_dir)
except:
    #print('out dir already exists!')
    pass
    
prefix = sys.argv[1]
prefix = prefix[prefix.rfind('/') + 1:prefix.rfind('.')]

boundaries = vdist > VTHRESH

z, p, v = rle(boundaries)

if z is None:
    print(z)
    print(boundaries)
    print(col_start, col_end)
#print('init', [x for x in zip(p, z)])
#print('*' * 20)
#print(p[v])

#z, p, v = filter(z, p, v, FILTER_THRESH)
#v = v != True
#print('filter', [x for x in zip(p, z)])
#print('*' * 20)

#z, p = merge(z, p, MERGE_THRESH)
def merge2(z, p):
    #print(z)
    n_z, n_p = [], []
    
    bad_z_idxs = z < EXPECTED_HEIGHT - EXPECTED_HEIGHT_TOLERANCE
    bz, bp, bv = rle(bad_z_idxs)
    for _z, _p in zip(bz, bp):
        if _z <= 3:
            for i in range(_z):
                n_z.append(z[_p + i])
                n_p.append(p[_p + i])
        else:
            n_z.append(z[_p])
            n_p.append(p[_p])
            n_p.append(p[_p + 1])

            cum_z = 0
            for i in range(1, _z - 1):
                cum_z += z[_p + i]
            n_z.append(cum_z)
            #print('converting', n_p[-1], cum_z)
            n_z.append(z[_p + _z - 1])
            n_p.append(p[_p + _z - 1])

    return np.array(n_z), np.array(n_p)

z, p = merge2(z, p)
#print('merge', [x for x in zip(p, z)])
#print('*' * 20)

z, p = split_erroneously_merged_rows(z, p)
#print('split', [x for x in zip(p, z)])
#print('*' * 20)
count = 0
out_count = 0
out_pixels = 0
for i, (_p, _z) in enumerate(zip(p, z)):
    #print(_p, _z)
    if _p > VERTICAL_START and _z >= EXPECTED_HEIGHT - EXPECTED_HEIGHT_TOLERANCE:
        count += 1
        #print('\t', out_pixels, count, out_count, i, _p, _z, p[i - 1])
        if out_pixels >= 215:
            #print('\toutput')
            start = p[i - 1] - TOP_CELL_BUFFER
            try:
                end = p[i + 1] + z[i + 1] + BOTTOM_CELL_BUFFER
            except:
                end = len(name)

            snippet = name[col_start + start: col_start + end, :]
            #print('\t', start, end)
            idx = str(out_count)
            if len(idx) == 1:
                idx = '0' + idx
            out_path = join(out_dir, prefix + '_' + idx + '.jpg')
            if len(snippet) > 75:
                print('WARNING: dims look weird', out_path, snippet.shape, start, end, _p, p[i - 1], p[i + 1], _z, z[i + 1])
            #print('outputting to', out_path)
            cv2.imwrite(out_path, snippet)
            out_count += 1
    out_pixels += _z
            
if out_count != 50:
    print("WARNING: output {} rows".format(out_count))
else:
    print("INFO: Correct number of rows output")
    #print(vdist)
#plt.scatter([i for i in range(len(vdist))], vdist)
#plt.plot(vdist)
#plt.plot(
#    x_data,
#    test_func(x_data, params[0], params[1]),
#    label='fitted function',
#    color = 'red'
#)
#plt.show()
#cv2.imwrite(out_path, name)
