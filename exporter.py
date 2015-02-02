import pandas as pd

def export_table_by_field(df, filename):
    rows = []
    columns = df.columns.values.tolist()
    for strcol in ('Soil_type', 'Soil_base_type'):
        if strcol in columns:
            columns.remove(strcol)

    for field, df_field in df[columns].groupby('Field_id'):
        rows.append(df_field.median().values)

    df_export = pd.DataFrame(rows, columns=columns)
    df_export.to_csv(filename, sep='\t', index=False)