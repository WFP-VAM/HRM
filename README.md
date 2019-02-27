# High Resolution Mapping of Food Security [![Build Status](https://travis-ci.org/WFP-VAM/HRM.svg?branch=master)](https://travis-ci.org/WFP-VAM/HRM)
for information on the project, please refer to the [GitHub page](https://wfp-vam.github.io/HRM/).

The application takes as input geo-referenced survey data, then for every survey _cluster_:
  - downloads relevant satellite images from the Google Maps Static API and Sentinel-2 from Google Earth Engine API. The class that handles this is the [img_lib.py](https://github.com/WFP-VAM/HRM/blob/master/Src/img_lib.py)
  - extract features from the images using neural networks trained [here](https://github.com/WFP-VAM/HRM_NN_Training). Class that handles it is the [nn_extractor.py](https://github.com/WFP-VAM/HRM/blob/master/Src/nn_extractor.py)
  - extract features as distance to hospital and school from OpenStreetMap using the [Overpass API](http://wiki.osm.org/wiki/Overpass_API). The [OSM_extactor.py[](https://github.com/WFP-VAM/HRM/blob/master/Src/osm.py) handles that part.
  - extract remote sensing indices from Sentinel 2, namely NDVI, NDBI and NDWI. [S2_indexes.py](https://github.com/WFP-VAM/HRM/blob/master/Src/rms_indexes.py) handles this part.
  - pull information from [ACLED](https://www.acleddata.com/) on violent events.
  - use ridge regression to infer indicator's value from the extarcted features. 
  
 All of the _training_ and _evaluaiton_ is coordinated by the [scripts/master.py](https://github.com/WFP-VAM/HRM/blob/master/scripts/master.py).
  
The trained models can then be used for making predictions in areas where no data is available. Use the [scripts/score_area.py](https://github.com/WFP-VAM/HRM/blob/master/scripts/score_area.py) for that. Work is in progress in the `application` directory for taking the method to produciton. 
  
### How to run the code:
#### File-system
Make sure to have the following file-system in place:
 ```
config
└── example_config.yaml 
Data
├── datasets
│   └── processed_survey.csv
├── Features
├── Geofiles
│   ├── ACLED
│   ├── NDs
│   ├── nightlights
│   ├── OSM
│   └── Rasters
│       └── base_layer.tif
└── Satellite   
    ├── Sentinel
    └── Google   
Models
  ```
 The mandatory files are: 
`Data/datasets/processed_survey.csv` this is your survey data! should contain at least 3 columns: "gpsLongitude","gpsLatitude" and one indicator. You can either work with individual survey data or aggregate the surveys to some geographic level. 
  
`Data/Geofiles/Rasters/base_layer.tif` is a raster file that containing the area of interest and the population density. Survey points will be snapped to its grid and the pulled layers over-laid.Please use 100x100m resolution WorldPop's rasters, available at http://www.worldpop.org.uk/data/data_sources/. 
 
`config/example_config` is the config file that you should fill in. Please use the template provided, fields list in there. 

#### Credentials
Because you will be pulling data from a number of sources, you will need to set up some credentials. 
##### Earth Engine
The most complex because it is using Google's OAuth2. Please follow [this link](https://developers.google.com/earth-engine/python_install_manual) to create your credentials file.
 
### Train Model
```
python master.py args 
```
where args is one or more `example_config.yaml`. Each `.yaml` should be space separated. Please run from the root directory of the application. For example to trigger for configs config_1.yaml, config_2.yaml and config_3.yaml do:
```
python master.py config_1.yaml config_2.yaml config_3.yaml > log.txt &
```
This will:
* download the relevant satellite images. (if not there already)
* extract the features for each image. (if no features for that id)
* pull and vectorize data from OSM, ACLED, Sentinel-2 and NOAA
* train the model on the existing data.
* write r2 Pearson scores to the ```Results/``` directory on a 5-fold cross validation loop.
* save the full predictions on the left-out data.
* save the trained model.

#### With Docker
Build with ``` docker build -t hrm . ``` then run with:
```
docker run -v ~/Desktop/HRM/HRM/Data:/app/Data -v ~/.config/earthengine:/root/.config/earthengine -t hrm ../config/example_config.yaml
```
First `-v` flag maps local directory `Data` to the same directory in the container. Second `-v` maps the earth engine credentials.
### Contacts
For more info to collaborate, use or just to know more reach us at jeanbaptiste.pasquier@wfp.org and lorenzo.riches@wfp.org or submit an issue.
