import os
from spanner import system


def retrieve_satellite_image_for_lat_lon_grid(df, filename, size=640):
    if os.path.exists(filename):
        return filename

    regions = []
    for lat, lon in set([tuple(row.tolist()) for row in df[['lat_int', 'lon_int']].values]):
        regions.append([lat - 0.5, lat + 0.5, lon - 0.5, lon + 0.5])

    return retrieve_satellite_image(regions, filename=filename, size=size)


def retrieve_satellite_image_for_region(df, filename, size=640):
    if os.path.exists(filename):
        return filename

    regions = []
    for field_id, df_field in df.groupby('Field_id'):
        regions.append(df_to_region(df_field, field_id))

    return retrieve_satellite_image(regions, filename=filename, size=size)


def retrieve_satellite_image_for_field(df, field_id, size=640):
    filename = os.path.join('data', 'satellite', '%s.png' % (field_id))
    if os.path.exists(filename):
        return filename

    return retrieve_satellite_image(df_to_region(df, field_id), filename, size=size)


def df_to_region(df, field_id):
    df_field = df[df['Field_id'] == field_id]
    lat_min = min(df_field['Latitude'])
    lat_max = max(df_field['Latitude'])
    lon_min = min(df_field['Longitude'])
    lon_max = max(df_field['Longitude'])
    return (lat_min, lat_max, lon_min, lon_max)


def retrieve_satellite_image(regions, filename, size=640):
    if not os.path.exists(filename):
        # google satellite image API
        api = ['https://maps.googleapis.com/maps/api/staticmap?&size={size}x{size}&maptype=hybrid'.format(size=size)]
        if len(regions) > 50:
            labelarg = 'markers={lat},{lon}'
            for r in regions:
                api.append(labelarg.format(lat=round(r[1], 2), lon=round(r[3], 2)))
                if len('&'.join(api)) > 2048:
                    print 'URL too long - truncating region labels!'
                    break
        else:
            patharg = 'path=color:0xff0000ff|weight:2|{latmn},{lonmn}|{latmn},{lonmx}|{latmx},'\
                      '{lonmx}|{latmx},{lonmn}|{latmn},{lonmn}'
            api.extend([patharg.format(latmn=r[0], latmx=r[1], lonmn=r[2], lonmx=r[3]) for r in regions])
        url = '&'.join(api)

        system.run_command('curl "%s" > %s' % (url, filename))

    return filename
