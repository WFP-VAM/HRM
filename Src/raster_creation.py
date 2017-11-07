import os
import yaml

with open('../public_config.yml', 'r') as cfgfile:
    public_config = yaml.load(cfgfile)

class NewRaster:
    """
    Class
    -----
    Create a raster (TIFF file) from the vector of predictions

    """

    def __init__(self,result_file,i="i",j="j",y_hat="y_hat",raster_file= \
    os.path.join("../Data","Satellite",public_config["satellite"]["grid"]),\
    image_dir=os.path.join("../Data","Satellite",public_config["satellite"]["source"])):
        from pandas import read_csv
        data=read_csv(result_file)
        #Scale the predictions and take the integer
        self.y_hat=(data[y_hat]*100).astype(int)
        self.i=data[i]
        self.j=data[j]
        (self.arr, self.cols, self.rows, self.ds)=self.__read_raster(raster_file)


    def __read_raster(self,raster_file):
        from gdal import Open
        ds = Open(raster_file)
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        [cols, rows] = arr.shape

        return (arr, cols, rows, ds)

    def new_grid(self,step=5):
        from numpy import zeros, int16
        arr_out = zeros((self.cols,self.rows), int16)

        for a in range (-step,step):
            for b in range (-step,step):
                arr_out[self.j+a,self.i+b]=self.y_hat
        return arr_out

    def write_save_raster(self,arr_out,name):
        from gdal import GetDriverByName, GDT_UInt16
        driver = GetDriverByName("GTiff")
        outdata = driver.Create("../Data/Results/{}.tif".format(name), self.rows, self.cols, 1, GDT_UInt16)
        outdata.SetGeoTransform(self.ds.GetGeoTransform())##sets same geotransform as input
        outdata.SetProjection(self.ds.GetProjection())##sets same projection as input
        outdata.GetRasterBand(1).WriteArray(arr_out)
        outdata.GetRasterBand(1).SetNoDataValue(0)
        outdata.FlushCache() #saves to disk
