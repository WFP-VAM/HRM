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

    import georasters as gr
    try:
        esa = gr.from_file(esa_raster)
    except MemoryError:
        print('Landuse Raster too big!')
        raise

    esa = esa.to_pandas()
    esa = esa[esa.value > 0]  # take only buildings

    # find the closest match between coordinates
    esa["gpsLatitude"] = esa.apply(lambda x: df[lat_col][abs(df[lat_col]-x["y"]).idxmin()],axis=1)
    esa["gpsLongitude"] = esa.apply(lambda x: df[lon_col][abs(df[lon_col]-x["x"]).idxmin()],axis=1)

    # merge on coordinates
    df = df.merge(esa, on=['gpsLongitude','gpsLatitude'])

    # TODO: use merge_asof faster, however only 1 column at time?
    # df.sort_values(['gpsLongitude','gpsLatitude'], inplace=True)
    # esa.sort_values(['x','y'], inplace=True)
    #
    # res = pd.merge_asof(df, esa, left_on=['gpsLongitude','gpsLatitude'], right_on=['x','y'] )

    return df