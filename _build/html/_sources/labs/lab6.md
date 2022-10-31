# Lab 6

````{caution}
This lab will provide you with an opportunity to practice exploratory spatial data analysis (ESDA), and specifically the local indicators of spatial association. The second part will focus on working through your Data Interim Report (due November 6, 2022). Please start by opening [this Google Colab notebook](https://colab.research.google.com/drive/1ebEcCaaKR_SWizV5hxtzCj26NDWNICAt?usp=sharing) and copying it to your Google Drive. 
````

````{important}
Please submit your lab by modifying the [template notebook](https://colab.research.google.com/drive/15D8bOdq6S4Ytn9kmIvdBNh_FxSePsDCk?usp=sharing)!
````

````{warning}
Make sure to submit the lab by **November 7 / 8, 2022**, depending on the scheduled time of your section.
````

---

## Part 1: Automating data ingestion with *gdown*.

See lab notebook for examples and instructions. 

## Part 2: Exploratory Data Analysis Workflow. Assessing AirBnb ads in Denver, CO. 

See lab notebook for examples and instructions. 

## Part 3: Working through Interim Report. 

See lab notebook for examples and instructions. 

---

## Lab instructions. 

> You will repeat the spatial autocorrelation analysis on AirBnb data for the place of your choosing. 

1. Go to the [insideairbnb website](http://insideairbnb.com/get-the-data/). 
2. Choose any metropolitan area! Download *listings.csv* and *neighbourhoods.geojson*. Rename if necessary.  

````{caution}
**You need to select urban / metro area, not entire state!**
````

````{tip}
select data with at least 30 neighborhoods which are connected by land (e.g. Hawaii would be trickier to analyze than Austin). 
````
3. Upload data to Google drive, import via gdown and unzip to the working directory. 
4. Calculate average price per neighborhood (unit of analysis in your *neighbourhood.geojson* file)
5. Join your calculated average to geographic data. 
6. Create a choropleth map using 'Fisher Jenks' scheme. 
7. Create a spatial weights matrix using Queens contiguity. Row-standardize the weights. 
8. Calculate and report global Moran's $I$ and simulated p-value at $\alpha=0.05$ confidence level. 
9. Plot both the histogram with simulated p-values and Moran's scatter plot 
10. Calculate and plot Local Moran's $I$ on a map at $\alpha=0.05$ level. 
11. Interpret your map characterizing spatial clusters and outliers. 
12. Submit to GauchoSpace as .ipynb and .pdf. 