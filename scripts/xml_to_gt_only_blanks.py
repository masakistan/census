import sys, json
from os.path import join
import xml.etree.ElementTree as ET
from collections import defaultdict

surname_if_blank = '---, '
xml_dir = sys.argv[2]
idx = 0

# NOTE: load manual corrections
corr = dict()
with open(sys.argv[3], 'r') as fh:
    for line in fh:
        if line[0] == '#':
            continue

        line = line.split('#')[0].strip()
        line = line.split('    ')
        key, cond, cons = line
        corr[key] = (cond, cons)
id = 566456
with open(sys.argv[1], 'r') as fh:
    for line in fh:
        page_id = line.strip()
        pid = page_id
        if len(page_id) == 0:
            continue
        xml_path = join(xml_dir, page_id + '.xml')
        root = ET.parse(xml_path).getroot()

        for type_tag in root.findall('headera/header-item'):
            if type_tag.get('name') == 'IMAGE_TYPE' and type_tag.text == 'No Extractable Data Image':
                for i in range(50):
                    s = "{}\tname_snippets/{}/{}_{}.jpg\t***".format(id, pid, pid, i)
                    print(s)
                    id += 1
