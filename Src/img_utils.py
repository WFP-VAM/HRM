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

def getRastervalue(row,raster_file):
    from osgeo import gdal,ogr
    import struct

    src_ds=gdal.Open(raster_file)
    gt=src_ds.GetGeoTransform()
    rb=src_ds.GetRasterBand(1)

    mx,my=row["gpsLongitude"],row["gpsLatitude"]  #coord in map units
    #Convert from map to pixel coordinates.
    #Only works for geotransforms with no rotation.
    px = int((mx - gt[0]) / gt[1]) #x pixel
    py = int((my - gt[3]) / gt[5]) #y pixel

    structval=rb.ReadRaster(px,py,1,1,buf_type=gdal.GDT_UInt16) #Assumes 16 bit int aka 'short'

    intval = struct.unpack('h' , structval) #use the 'short' format code (2 bytes) not int (4 bytes)
    return intval[0] #intval is a tuple, length=1 as we only asked for 1 pixel value
