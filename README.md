# High Resolution Mapping of Food Security

To understand the food security situation of vulnerable populations, the World Food Programme is continuously conducting household surveys. The difficulties of collecting face-to-face data in remote or unsafe areas mean that the estimates are only representative at a low resolution - usually regional or district level aggregation - for a reasonable cost. However, WFP and other humanitarian actors need more detailed maps in order to allocate resources more efficiently. At the same time, geographic coordinates are increasingly being recorded in household surveys thanks to digital data collection tools. 

The main aim of our initiative is to leverage the high resolution mapping techniques developed in academia for use in WFP and  the humanitarian sector, and to make it accessible for a broad range of users. Another objective is to get the most accurate fine-scale maps on food security indicators. Our work was mostly inspired by the [WorldPop/Flowminder “bottom-up” approach to population mapping](http://www.worldpop.org.uk/about_our_work/case_studies/) and by [transfer learning techniques developed by Stanford University to poverty mapping](http://science.sciencemag.org/content/353/6301/790).

The code-base is able to:
  - read in geo-referenced survey data.
  - downlaod relevant satellite images from a number of sources (Google Maps, Bing Maps, Sentinel).
  - extract features from the images using pre-trained neural netowrks.
  - use ridge regression to infer indicator's value from the extarcted features. 
  
The trained models can then be used for making predicitons in areas where no data is available. Work is in progress in the `application` directory for taking the method to produciton. 
  
### How to run the code:
#### Set-up
Set up the necessary parameters in the ```config``` table. (WIP) and copy your dataset to the ```Data/datasets``` folder. Don't forget to populate the table with information about:
* The dataset you are using and the filename. Check the data structure to see where to save the survey data.
* The source of high-resolution satellite imageries, the number of tiles per point you are using and the tiff file you are using to define your gris coordinate system. Check the data structure to see where to save this tiff file.
* The CNN network and the layer you are using to extract features.

Don't forget to populate the private_config.yml with information about:
* You bing and/or API keys 

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
  
`raster_of_the_world.tif` is a raster file that associated GRID coordinates to the areas of interest. We are currently working with a global raster at a 1km resolution taken from the NOAA nightlights "F182013.v4c_web.stable_lights.avg_vis.tif" and available at https://ngdc.noaa.gov/eog/dmsp/downloadV4composites.html. This raster can also be used to fine-tune the CNN.
 
 ### Work in progress
  
 The next steps are (work in progress):
+ App development (can be found in the `application` directory.
+ Better fine tuning for each data source. 
+ Validation on more datasets and usecases.
+ Adding features coming from other sources 
  + Ex: conflict GIS maps or CDR data (WorldPop approach) 
  
### Ridge Regression
![coefficientsa as a function of the L2 parameter Alpha for VGG16 features](https://github.com/WFP-VAM/HRM/blob/master/Plots/alpha_for_VGG16_features.png)

### Contacts
For more info to collaborate, use or just to know more reach us at jeanbaptiste.pasquier@wfp.org and lorenzo.riches@wfp.org or submit an issue.
