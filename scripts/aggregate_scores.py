import os
import sys
try:
    os.chdir('scripts')
except FileNotFoundError:
    pass
sys.path.append(os.path.join("..", "Src"))
from utils import weighted_sum_by_polygon
import click


# ---------- #
# PARAMETERS #
@click.command()
@click.option('--config_id')
@click.option('--shapefile', help="path to the input shapefile in a format support by geopandas")
@click.option('--pop_weights', help="path to the raster containing the weights (ex: population data)")
def main(config_id, shapefile, pop_weights):

    input_rst = "../Data/Results/scalerout_{}.tif".format(config_id)
    output_shp = "../Data/Results/scalerout_{}_aggregated.shp".format(config_id)

    weighted_sum_by_polygon(shapefile, input_rst, pop_weights, output_shp)


if __name__ == '__main__':

    main()
