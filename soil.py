import os
import re
from spanner import flatfile

soiltype_to_properties = flatfile.load_dict(os.path.join('data', 'soil_properties.txt'),
                                            ['type'], ['clay', 'sand', 'silt'])

def composition(soiltype):
    characteristics = {'sand', 'clay', 'silt', 'loam'}
    adjectives = {'sandy', 'silty', 'loamy'}
    keywords = adjectives.union(characteristics)
    words = [re.sub(r's$', '', w.lower()) for w in soiltype.split() if re.sub(r's$', '', w.lower()) in keywords]

    basetype = ' '.join(words)
    if basetype in soiltype_to_properties:
        return basetype, soiltype_to_properties[basetype]
    else:
        # print soiltype, words, 'Uknown type'
        return None, (None, None, None)


