# Data Report

These report guidelines will help you get started on your geographical analysis project. The key to successful project is good data. There are numerous data repositories online and even more interesting questions that you can ask and answer with the data. 

````{warning}
**The Data Report is due 10/16/2022 by 23.59.** 
````

---

## Data sources 

Please see below for an outline of available data sources: 

1. [US Census Bureau](). Consider using CenPy API for Python (which makes getting data from Census much easier)
2. [Kaggle](). Various data for data science and machine learning applications. 
3. [Free GIS Data](https://freegisdata.rtwilson.com/). Contains list of useful resources to GIS data. 
4. [ESRI Open Data Hub](http://opendata.arcgis.com/)
5. [OpenStreetMap](https://gisgeography.com/openstreetmap-download-osm-data/). Please see demo from Lecture 1 on how to utilize Python package *osmnx* to get the OSM data out programmatically. 
6. [GIS Geography List for Data](https://gisgeography.com/best-free-gis-data-sources-raster-vector/)
7. [GIS data by country from Wikipedia](https://en.wikipedia.org/wiki/List_of_GIS_data_sources)
8. [Berkeley GIS International Data Library](https://guides.lib.berkeley.edu/gis/international). This library contains links to resources broadly grouped into the following categories: *basic GIS data, GIS data for humanities, GIS data for natural sciences, GIS data for social sciences, California and Berkeley GIS data*.  
9. [ArcGIS Living Atlas of the World](https://livingatlas.arcgis.com/en/browse/)
10. [National historical geographic information systems (NHGIS)](https://www.nhgis.org/). easy access to summary tables and time series of population, housing, agriculture, and economic data, along with GIS-compatible boundary files, for years from 1790 through the present and for all levels of U.S. census geography, including states, counties, tracts, and blocks.

---

## Data report requirements 

This is a course on geographical analysis. Thus, the data that you are analyzing should be geographically-referenced. For the purposes of this report you will need to download at least two data sets: spatial/spatio-temporal data and supplementary data. 

### Spatial and Spatio-temporal data 

* Points, lines and polygons. 
* File-format: GeoPackage (.gpkg), ShapeFile (.shp), GeoDatabase (.gdb), GeoJSON (.geojson), etc. 
* Samples size of at least 30 observations (BUT more is better!). Aim for hundreds, thousands, tens of thousands. 
* For data that has a temporal component (datetime), there are many more ways to summarize it. This will make it easier to ask questions, but will also complicate your data analysis. 

````{tip}
**Examples of spatial variables (polygons)**: *average rent in LA neighborhoods*, *percent of renter-occupied units in LA Census block groups*, *medium income in Census tracts*, *COVID-19 cases in US counties*. 

**Examples of spatial variables (points)**: *AirBnB ads in New York City*, *Average temperature from weather station data*, *bike sharing stations data*, *locations of coffee shops in LA* 

**Examples of spatial variables (lines)**: *US rivers*, *US roads*, *GPS tracks of animals*. 

**Examples of supplementary data**: *Census block groups*, *Census blocks*, *Census tracts*, *LA neighborhoods*, *NYC boroughs*. 
````

### Python functions to practice 

* reading data (pd.read_csv, gpd.read_file) 
* filtering data (df = df.loc[df.var1=='lab1'])
* selecting variables (df = df[['var1', 'var2']])
* calculating new variables (df['new_var'] = df.old_var + df.old_var2)
* summarizing data (df.describe)
* plot data (df.plot, gdf.plot)


---

## Data report instructions 

1. Locate and download the data of your interest. 
2. Upload your data into Google Colab notebook (or other analytical software of your choice). 
3. Import uploaded data into Pandas/GeoPandas dataframe. 
4. Summarize at least two categorical and/or continuous variable in your dataset using 'describe' function. 
5. Create a histogram for at least two continuous variable in your data. Describe the distribution on each plot (unimodal, multimodal, normal, exponential, logarithmic, skewed, etc.). 
6. Create a geographic map of your study area. Plot your spatial/spatio-temporal data and supplemental/reference data. 
7. Write at least three interesting questions that you intend to study in your geographical analysis. 
8. Submit to GauchoSpace as *ipynb notebook or pdf. 

