def get_cell_idx(lon, lat, top_left_x_coords, top_left_y_coords):
    """
    Function
    --------
    get_cell_idx

    Given a point location and all the pixel locations of the raster file,
    get the column and row index of the point in the raster

    Parameters
    ----------
    lon : float
        Longitude of the point
    lat : float
        Latitude of the point
    top_left_x_coords : numpy.ndarray  shape: (number of columns,)
        Longitude of the top-left point in each pixel
    top_left_y_coords : numpy.ndarray  shape: (number of rows,)
        Latitude of the top-left point in each pixel

    Returns
    -------
    lon_idx : int
        Column index
    lat_idx : int
        Row index
    """
    import numpy as np

    lon_idx = np.where(top_left_x_coords < lon)[0][-1]
    lat_idx = np.where(top_left_y_coords > lat)[0][-1]
    return lon_idx, lat_idx


def getRastervalue(df, esa_raster, lat_col="gpsLatitude", lon_col="gpsLongitude"):
    """
    when you pass dataframe with Lat, Long coordinates
    it returns a vector of the corresponding land use value at theses locations

    It merges on the closest coordinates between the raster and the dataset.

    For now is focused on the ESA landuse raster.

    use: data = getRastervalue(data,path_to_raster)
    """

    print('-> finding landuse for {} points'.format(df.shape[0]))

    import georasters as gr
    try:
        esa = gr.load_tiff(esa_raster)
    except MemoryError:
        print('Landuse Raster too big!')
        raise

    # Find location of point (x,y) on raster, e.g. to extract info at that location
    NDV, xsize, ysize, GeoT, Projection, DataType = gr.get_geo_info(esa_raster)

    def lu_extract(row):

        try:
            c, r = gr.map_pixel(row[lon_col], row[lat_col], GeoT[1], GeoT[-1], GeoT[0], GeoT[3])
            lu = esa[c, r]
            return lu

        except IndexError:
            pass

    df['landuse'] = df.apply(lu_extract, axis=1)

    print(df.landuse.unique())
    df = df[df.landuse == 1]

    return df
