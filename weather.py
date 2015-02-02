import os
import re
import numpy as np
import pandas as pd
from spanner.decorators import validate_args
from spanner import system


TBASE = 10

@validate_args(int, int)
def retrieve_weather_data(lat, lon):
    filename = os.path.join('data', 'weather', '%d_%d_weather.txt' % (lat, lon))
    if not os.path.exists(filename):
        api = 'http://power.larc.nasa.gov/cgi-bin/cgiwrap/solar/agro.cgi?email=agro%40larc.nasa.gov'\
              '&step=1&lat={lat}&lon={lon}&ms=1&ds=1&ys=2013&me=12&de=31&ye=2013&submit=Yes'
        url = api.format(lat=lat, lon=lon)
        data, stderr = system.run_command('curl "%s"' % url)
        with open(filename, 'w') as ofh:
            ofh.write(data)

    return load_from_disk(filename)


def load_from_disk(filename):
    with open(filename, 'r') as fh:
        data = fh.read()

    rows = []
    for line in data.split('\n'):
        if line.startswith('@ WEYR'):
            header_cols = re.split(r'\s+', line[2:])

        elif re.search(r"^\s+[0-9]{4,}", line):
            rows.append([float(x.strip()) for x in re.split(r'\s+', line) if x.strip() != ''])

    df = pd.DataFrame(rows, columns=header_cols)
    df.set_index('WEDAY')
    df['GDD'] = (df['TMAX'] + df['TMIN']) / 2 - TBASE
    df['GDD'] = df['GDD'].map(lambda x: max(0, x))
    df['GDD_cumulative'] = np.cumsum(df['GDD'].tolist())

    return df

