# High Resolution Mapping of Food Security

for information on the project, please refer to the [GitHub page](https://wfp-vam.github.io/HRM/).

The application takes as input geo-referenced survey data, then for every survey _cluster_:
  - downlaods relevant satellite images from the Google Maps Static API and Sentinel-2 from Google Earth Engine API. THe class that handles this is the [img_lib.py](https://github.com/WFP-VAM/HRM/blob/master/Src/img_lib.py)
  - extract features from the images using neural netowrks trained [here](https://github.com/WFP-VAM/HRM_NN_Training). Class that handles it is the [nn_extractor.py](https://github.com/WFP-VAM/HRM/blob/master/Src/nn_extractor.py)
  - extrat features as distance to hospital and school from OpenStreetMap using the [Overpass API](http://wiki.osm.org/wiki/Overpass_API). The [OSM_extactor.py[](https://github.com/WFP-VAM/HRM/blob/master/Src/osm.py) handles that part.
  - extarct remote sening indices from Sentinel 2, namely NDVI, NDBI and NDWI. [S2_indexes.py](https://github.com/WFP-VAM/HRM/blob/master/Src/rms_indexes.py) handles this part.
  - use ridge regression to infer indicator's value from the extarcted features. 
  
 All of the _training_ and _evaluaiton_ is coordinated by the [scripts/master.py](https://github.com/WFP-VAM/HRM/blob/master/scripts/master.py).
  
The trained models can then be used for making predicitons in areas where no data is available. USe the [scripts/score_area.py](https://github.com/WFP-VAM/HRM/blob/master/scripts/score_area.py) for that. Work is in progress in the `application` directory for taking the method to produciton. 
  
### How to run the code:
#### Set-up
Set up the necessary parameters in the ```config``` database table and configure the necessary DB paramters filling _config_template.yml_ and renaming it to _config_private.yml_ (sorry messy, is a WIP) and copy your dataset to the ```Data/datasets``` folder. Don't forget to populate the table with information about:
* The dataset you are using and the filename. Check the data structure to see where to save the survey data.
* The source of high-resolution satellite imageries, the number of tiles per point you are using and the tiff file you are using to define your gris coordinate system. Check the data structure to see where to save this tiff file.
* Aggregation parameters


#### Train Model
```
python master.py args
```
where args are the ids of the table "config" in the database. Each id should be space sperated, example to trigger for configs 1, 2 and 3 do:
```
python master.py 1 2 3 > log.txt &
```
This will:
* downlaod the relevant satellite images. (if not therte already)
* extract the features for each image. (if no features for that config_id)
* write r2 Pearson and MAPE scores to the ```results``` table.
* save the full predictions on the left-out data.
* save the trained model.

  
### Data Structure
  
 ```
 
Data
├── datasets
│   └── processed_survey.csv
├── Features
│   └── features_config_id_1.csv
└── Satellite
    ├── raster_of_the_world.tif
    ├── Bing
    │   └── 5.166666_13.34166_16.jpg
    └── Google
    │   └── 5.166666_13.34166_16.jpg
    ├── Sentinel
    │   └── 5.16666_13.34166_2016-01-01_2017-01-01.jpg
  ```
  
`processed_survey.csv` should contain at least 3 columns: "gpsLongitude","gpsLatitude" and one indicator. You can either work with individual survey data or aggregate the surveys to some geographic level. 
  
`raster_of_the_world.tif` is a raster file that associated GRID coordinates to the areas of interest. We are currently working with WorlPop rasters starting at 100x100m resolution, available at http://www.worldpop.org.uk/data/data_sources/. 
 
 ### Work in progress
  
 The next steps are (work in progress):
+ App development (can be found in the `application` directory.
+ Open Source, making it easy to use for YOU!
+ Validation on more datasets and usecases.
+ Adding features coming from other sources 
  + Ex: conflict GIS maps or CDR data (WorldPop approach) 
  
### Ridge Regression
![coefficientsa as a function of the L2 parameter Alpha for VGG16 features](https://github.com/WFP-VAM/HRM/blob/master/Plots/alpha_for_VGG16_features.png)

### Contacts
For more info to collaborate, use or just to know more reach us at jeanbaptiste.pasquier@wfp.org and lorenzo.riches@wfp.org or submit an issue.
