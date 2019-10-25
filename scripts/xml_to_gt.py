import sys, json
from os.path import join
import xml.etree.ElementTree as ET
from collections import defaultdict


xml_dir = sys.argv[2]
idx = 0
with open(sys.argv[1], 'r') as fh:
    for line in fh:
        xml_path = join(xml_dir, line.strip() + '.xml')
        root = ET.parse(xml_path).getroot()

        i = 0
        surn = None
        gn = None

        phid = None
        set_hid_to_surn = False
        #print('xml path:', xml_path)
        skipped = set()
        for type_tag in root.findall('headera/header-item'):
            #print(i)
            #print(type_tag.get('name'))

            #NOTE: check if blank row
            if type_tag.get('record') in skipped:
                continue
            
            if type_tag.get('name') == 'PR_NAME_SURN' and type_tag.text is not None:
                for sub_tag in type_tag.findall('md/md'):
                    if sub_tag.get('name') == 'marker' and sub_tag.get('value') == 'Blank':
                        #print('incrementing blank')
                        i += 1
                        skipped.add(type_tag.get('record'))
                        continue
                surn = type_tag.text.strip()
                if set_hid_to_surn:
                    hid = surn
                    set_hid_to_surn = False
            elif type_tag.get('name') == 'PR_NAME_GN' and type_tag.text is not None:
                gn = type_tag.text.strip()
            elif type_tag.get('name') == 'HOUSEHOLD_ID' and type_tag.text is not None:
                hid = type_tag.text.strip()
                if len(hid) == 0:
                    set_hid_to_surn = True

            if surn is not None and gn is not None and (len(surn) > 0 and len(gn) > 0):
                info = join('name_snippets', line.strip(), line.strip() + '_' + str(i) + '.jpg')
                assert i < 60, info

                #print(phid, hid)
                if phid != hid:
                    print('\t'.join(map(str, [idx, info, surn + ', ' + gn])))
                else:
                    print('\t'.join(map(str, [idx, info, gn])))

                phid = hid
                surn = None
                gn = None
                i += 1
                idx += 1
        
