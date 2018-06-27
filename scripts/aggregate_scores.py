import os
import sys
try:
    os.chdir('scripts')
except FileNotFoundError:
    pass
sys.path.append(os.path.join("..", "Src"))
from sqlalchemy import create_engine
import yaml
import pandas as pd
from utils import weighted_sum_by_polygon
import click


# ---------- #
# PARAMETERS #
@click.command()
@click.option('--config_id')
@click.option('--shapefile', default=("../Data/Geofiles/Shapefiles/ADM2/bfa_admbnda_adm2_1m_salb_itos/bfa_admbnda_adm2_1m_salb_itos.shp"))
def main(config_id, shapefile):

    with open('../private_config.yml', 'r') as cfgfile:
        private_config = yaml.load(cfgfile)

    # connect to db and read config table
    engine = create_engine("""postgresql+psycopg2://{}:{}@{}/{}"""
                           .format(private_config['DB']['user'], private_config['DB']['password'],
                                   private_config['DB']['host'], private_config['DB']['database']))

    input_shp = shapefile
    input_rst = "../Data/Results/scalerout_{}.tif".format(config_id)
    weight_rst = "../tmp/final_raster.tif"  #TODO: clip config["satellite_grid"][0] to extend of results

    output_shp = "../Data/Results/scalerout_{}_aggregated.shp".format(config_id)


    weighted_sum_by_polygon(input_shp, input_rst, weight_rst, output_shp)

if __name__ == '__main__':

    main()
