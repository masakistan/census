import sys, json
import numpy as np


with open(sys.argv[1], 'r') as fh:
    d = json.load(fh)

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

for shape in d['shapes']:
    if shape['shape_type'] != 'polygon':
        continue
    points = shape['points']
    label = shape['label']
    x = [x for x, y in points]
    y = [y for x, y in points]
    #print(points)
    #print(x)
    #print(y)
    #print(label, PolyArea(x, y))
    a = PolyArea(x, y)
    if a < 10000:
        #print(label, a)
        print(sys.argv[1])
        sys.exit()
