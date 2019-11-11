import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from os.path import join
from os import makedirs

START, END = 280, 735
THRESH = 160
VTHRESH = 60000 #60000
HEADER_START_THRESH = 50
EXPECTED_HEADER_HEIGHT = 130
PAD = 0
MERGE_THRESH = 1
VERTICAL_START = 0
VERTICAL_END = 2450
NAME_COLUMN_MIN_WIDTH = 300

FILTER_THRESH = 3
EXPECTED_HEIGHT = 35
EXPECTED_HEIGHT_TOLERANCE = 10
MERGED_ROW_TOLERANCE = 10
COL_END_THRESH = 100
COL_START_THRESH = 500
TOP_CELL_BUFFER = 2
BOTTOM_CELL_BUFFER = 20

BIN_THRESH = 220

ADAPTIVE_WINSZ = 11
CCOLORS = [
    (255, 0, 0),
    (255, 63, 0),
    (255, 127, 0),
    (255, 191, 0),
    (255, 255, 0),
    (191, 255, 0),
    (127, 255, 0),
    (63, 255, 0),
    (0, 255, 0),
    (0, 255, 63),
    (0, 255, 127),
    (0, 255, 191),
    (0, 255, 255),
    (0, 191, 255),
    (0, 127, 255),
    (0, 63, 255),
    (0, 0, 255),
    (63, 0, 255),
    (127, 0, 255),
    (191, 0, 255),
    (255, 0, 255),
    (255, 0, 191),
    (255, 0, 127),
    (255, 0, 63),
]


def fltp(point):
    return tuple(point.astype(int).flatten())


class ContourInfo(object):

    def __init__(self, contour, rect, mask):

        self.contour = contour
        self.rect = rect
        self.mask = mask

        self.center, self.tangent = blob_mean_and_tangent(contour)

        self.angle = np.arctan2(self.tangent[1], self.tangent[0])

        clx = [self.proj_x(point) for point in contour]

        lxmin = min(clx)
        lxmax = max(clx)

        self.local_xrng = (lxmin, lxmax)

        self.point0 = self.center + self.tangent * lxmin
        self.point1 = self.center + self.tangent * lxmax

        self.pred = None
        self.succ = None

    def proj_x(self, point):
        return np.dot(self.tangent, point.flatten()-self.center)

    def local_overlap(self, other):
        xmin = self.proj_x(other.point0)
        xmax = self.proj_x(other.point1)
        return interval_measure_overlap(self.local_xrng, (xmin, xmax))

def blob_mean_and_tangent(contour):

    moments = cv2.moments(contour)

    area = moments['m00']
    #print('area', area)
    #print(contour)
    #print(moments['m10'])
    #print(moments['m01'])

    if area < 0.00001:
        mean_x = 0
        mean_y = 0
    else:
        mean_x = moments['m10'] / area
        mean_y = moments['m01'] / area

    moments_matrix = np.array([
        [moments['mu20'], moments['mu11']],
        [moments['mu11'], moments['mu02']]
    ]) / area

    _, svd_u, _ = cv2.SVDecomp(moments_matrix)

    center = np.array([mean_x, mean_y])
    tangent = svd_u[:, 0].flatten().copy()

    return center, tangent


def box(width, height):
    return np.ones((height, width), dtype=np.uint8)


def make_tight_mask(contour, xmin, ymin, width, height):

    tight_mask = np.zeros((height, width), dtype=np.uint8)
    tight_contour = contour - np.array((xmin, ymin)).reshape((-1, 1, 2))

    cv2.drawContours(tight_mask, [tight_contour], 0,
                     (1, 1, 1), -1)

    return tight_mask


def debug_show(name, step, text, display):
    filetext = text.replace(' ', '_')
    outfile = name + '_debug_' + str(step) + '_' + filetext + '.png'
    cv2.imwrite(outfile, display)

def visualize_contours(name, small, cinfo_list):

    regions = np.zeros_like(small)

    for j, cinfo in enumerate(cinfo_list):

        cv2.drawContours(regions, [cinfo.contour], 0,
                         CCOLORS[j % len(CCOLORS)], -1)

    mask = (regions.max(axis=2) != 0)

    display = small.copy()
    display[mask] = (display[mask]/2) + (regions[mask]/2)

    for j, cinfo in enumerate(cinfo_list):
        color = CCOLORS[j % len(CCOLORS)]
        color = tuple([c/4 for c in color])

        cv2.circle(display, fltp(cinfo.center), 3,
                   (255, 255, 255), 1, cv2.LINE_AA)

        #print(cinfo.point0, cinfo.point1)
        cv2.line(display, fltp(cinfo.point0), fltp(cinfo.point1),
                 (255, 255, 255), 1, cv2.LINE_AA)

    debug_show(name, 1, 'contours', display)


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
        #print(_p, _z)
        if _z > EXPECTED_HEIGHT + MERGED_ROW_TOLERANCE:
            print('splitting!', _z % EXPECTED_HEIGHT, EXPECTED_HEIGHT - MERGED_ROW_TOLERANCE)
            #if _z % EXPECTED_HEIGHT < MERGED_ROW_TOLERANCE:
            factor = _z // EXPECTED_HEIGHT
            if _z % EXPECTED_HEIGHT > EXPECTED_HEIGHT - (1.5 * MERGED_ROW_TOLERANCE):
                #factor = (_z // EXPECTED_HEIGHT) + 1
                factor += 1
            #print("\tfound row to be split", _p, _z, 'factor:', factor)
            #print(factor)
            print(factor)
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


# NOTE: names should be somewhere in here
img_bin = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 11)
hdist = np.mean(img_bin, axis = 0)

#plt.plot([i for i in range(len(hdist))], hdist)
#plt.show()

boundaries = hdist > THRESH
z, p, v = rle(boundaries)
z = z[v]
p = p[v]
cand_idx = z > NAME_COLUMN_MIN_WIDTH
z = z[cand_idx]
p = p[cand_idx]
#print(p[0], p[0] + z[0])

#print(p)
start = p[0]
end = p[0] + z[0]
#print(start, end)

#assert end - start > NAME_COLUMN_MIN_WIDTH, "error! name column is too small"

name = img[:, start - PAD : end + PAD]
name_bin = img_bin[:, start - PAD : end + PAD]
cv2.imwrite('col_1_before.jpg', name_bin)

#cv2.imwrite('col_1_before.jpg', name_bin)

vdist = np.mean(name_bin, axis = 1)
cv2.imwrite('frag.jpg', name_bin[:500])
vdist = list(reversed(vdist[:500]))

plt.figure(figsize=(1,8), dpi = 600)
plt.yticks([])
plt.xticks([])
plt.plot(vdist, [i for i in range(len(vdist))])
plt.savefig('fig.jpg')
#plt.show()
print('\n'.join(map(str, vdist)))

col_boundaries = vdist > HEADER_START_THRESH
z, p, v = rle(col_boundaries)
z = z[v]
p = p[v]
cand_idx = z > EXPECTED_HEADER_HEIGHT
z = z[cand_idx]
p = p[cand_idx]

start = p[0] + z[0]
name = name[start:]
name_bin = name_bin[start:]
cv2.imwrite('col_2_header removed.jpg', name_bin)

#print(start)

#print(cp, cv)
vdist = np.mean(name_bin, axis = 1)
#plt.plot([i for i in range(len(vdist))], vdist)
#plt.show()

col_boundaries = vdist < HEADER_START_THRESH
z, p, v = rle(col_boundaries)
p = p[v]
#print(cp)
if len(p) == 0:
    col_start = 0
    col_end = len(name_bin)
else:
    col_start = p[0]
    col_end = p[-1]

    if col_start > COL_START_THRESH:
        col_start = 200

    if col_end < len(name_bin) - COL_END_THRESH:
        col_end = len(name_bin)
#print(header_start)
name_bin = name_bin[col_start:col_end]
name = name[col_start:col_end]

col_start += start
#print(col_start, col_end)
cv2.imwrite('col_3_fixed.jpg', name)
vdist = np.sum(name_bin, axis = 1)

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
    print("ERROR: could not process {}".format(sys.argv[1]))

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
for i, (_p, _z) in enumerate(zip(p, z)):
    #print(_p, _z)
    if _z >= EXPECTED_HEIGHT - EXPECTED_HEIGHT_TOLERANCE:
        count += 1
        #print('\t', count, out_count, i, _p, _z, p[i - 1])
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
