import os
import re
import numpy as np
import pandas as pd
from sklearn import cluster
import weather
import soil
import exporter


def load_data_frame(filename):
    with open(filename, 'r') as fh:
        data = fh.read()

    # Data file had extra line feeds after soil type column
    data = re.sub(r'([0-9].*)([a-z])\n', r'\1\2', data)
    lines = [line.split(',') for line in data.split('\n')]
    df = pd.DataFrame(lines[1:], columns=lines[0])


    # cast to appropriate datatypes
    col_to_datatype = {'Field_id': int,
                        'Longitude': float,
                        'Latitude': float,
                        'WDRVI': float,
                        'DOY': int,
                        'Soil_type': str,
                        'Slope': float,
                        'Aspect': float,
                        'Curvature': float}

    for column in col_to_datatype:
        df[column] = df[column].map(lambda x: col_to_datatype[column](x))


    # break soil type down into percent clay/sand/silt
    soiltype_to_composition = dict()
    soiltype_to_basetype = dict()
    for soiltype in set(df['Soil_type']):
        basetype, composition = soil.composition(soiltype)
        soiltype_to_basetype[soiltype] = basetype
        soiltype_to_composition[soiltype] = composition

    df['Soil_base_type'] = df['Soil_type'].map(lambda x: soiltype_to_basetype[x])
    for idx, colname in enumerate(['Pct_clay', 'Pct_sand', 'Pct_silt']):
        df[colname] = df['Soil_type'].map(lambda x: soiltype_to_composition[x][idx])

    # drop unknown soil types
    before = len(df)
    df = df.dropna(subset=['Soil_base_type'])
    if len(df) != before:
        print 'Dropped %d of %d rows due to unknown soil type' % (before - len(df), before)


    # calculate leaf area index
    df['LAI'] = df['WDRVI'].map(lambda x: np.log((1/(1-((x+0.6684)/1.4392))))/0.3418)


    # set aspect 0 = south so we can use a diverging colormap centered at 0 when printing
    df['Aspect'] = df['Aspect'].map(lambda x: x - 180)

    # create slope * aspect column - aspect shouldn't matter unless slope is non-zero
    df['Slope_x_aspect'] = df['Slope'] * df['Aspect']

    # retrieve weather data from NASA
    df['lat_int'] = df['Latitude'].map(lambda x: int(round(x)))
    df['lon_int'] = df['Longitude'].map(lambda x: int(round(x)))

    # put weather dataframes in a dictionary accessed by integer latitude/longitude
    # (the resolution of data available)
    latlon_to_weather = dict()
    for lat, lon in set([tuple(row.tolist()) for row in df[['lat_int', 'lon_int']].values]):
        latlon_to_weather[(lat, lon)] = weather.retrieve_weather_data(lat, lon)

    # calculate per-field GDD
    # assumes any given field does not cross integer lat/lon boundaries
    field_to_gdd = dict()
    for field_id, df_field in df.groupby('Field_id'):
        lat_int, lon_int, doy = df_field[['lat_int', 'lon_int', 'DOY']].values[0].tolist()
        df_weather = latlon_to_weather[(lat_int, lon_int)]
        field_to_gdd[field_id] = df_weather[df_weather['WEDAY'] == doy]['GDD_cumulative'].values[0]

    df['GDD'] = df['Field_id'].map(lambda x: field_to_gdd[x])


    # cluster locations
    kmeans = cluster.KMeans(init='k-means++', n_clusters=8, n_init=100)
    positions = df[['Latitude', 'Longitude']].values
    kmeans.fit(positions)
    df['Region_id'] = kmeans.predict(positions)

    # export field-level summary data
    exporter.export_table_by_field(df, os.path.join('data', 'field_data.txt'))

    # make a guess about crop type for each field
    field_to_crop_guess = dict()
    for field_id, df_field in df.groupby('Field_id'):
        field_to_crop_guess[field_id] = 'soybeans' if np.mean(df_field['LAI'].values) < 3 else 'corn'

    df['Crop_guess'] = df['Field_id'].map(lambda x: field_to_crop_guess[x])
    # df['Crop_guess'] = df['LAI'].map(lambda x: 'soybeans' if x < 3 else 'corn')

    return df