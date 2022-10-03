# Lab 2 - Summarizing Data and Basic Python Plotting

````{caution}
Please use this webpage as a set of general recommendations. The lab itself is located in [Google Colab](https://colab.research.google.com/drive/1Ggm6WsRVj95zx8AemRPu49Qk5Ecm503d?usp=sharing). Make sure to copy the notebook as you work through it. 
````

````{warning}
Please make sure to submit the lab by **October 10/11, 2022**, depending on your section. 
````

This lab will teach you how to work import, export and analyze data in Python, and specifically cover the following concepts: 

1. Working with data
    1. Importing and saving data with Pandas 
    2. Importing and saving data with GeoPandas 
    3. Converting between Pandas and Geopandas 
2. Calculating measures of location and variation 
3. Creating basic plots and charts in Matplotlib and Seaborn
4. Getting data from OSM and from CenPy 

---

Optional useful tutorials on lab topics: 

1. [**GeoPandas** Getting Started](https://geopandas.org/en/stable/getting_started/introduction.html)
2. [10 minutes to **Pandas**](https://pandas.pydata.org/docs/user_guide/10min.html)
3. [**Matplotlib** Tutorial](https://matplotlib.org/stable/tutorials/introductory/pyplot.html)
4. [Intro to **Seaborn**](https://seaborn.pydata.org/tutorial/introduction.html)
5. [PythonGIS Intro to data analysis](https://pythongis.org/part1/chapter-03/index.html)
6. [PythonGIS Intro to data visualization](https://pythongis.org/part1/chapter-04/index.html)

--- 

## Working with data 

### Reading the data

Rectangular data in Python is typically handled via Pandas. This includes files with the following extensions: .csv, .xls and .xlsx (Excel), .tab (tab delimited), .txt (text format), and many more.  

```python
import pandas as pd

df = pd.read_csv('path_to_file') 
```

In Pandas, the data is typically stored in two-dimensional DataFrame objects with (un)labeled rows and columns. Another way to think about Pandas DataFrames is by analogy with programmable spreadsheets. A one-dimensional array (any one variable) is called Pandas Series. 

````{tip}
When working with data in languages other than English, consider using *Unicode Transformation Format - 8 bit* or **UTF-8** for short. Most *read* functions in both Python and R packages provide interface to set encoding when reading/writing information onto disk. 
````

```python
# import data specifying the encoding 
df = pd.read_csv('path_fo_file', encoding='utf-8')
```

````{note}
Computers store information in binary system as sequence of 0 and 1. Because text is made of individual characters it is represented by a string of bits. For example, letter 'A' is represented as '01000001' and letter 'a' is represented as '01100001'. Normally, a standardized encoding system is used to translate from byte code to characters. One early example of such system is American Standard Code for Information Interchange (ASCII), but it only works for latin alphabet, assigning a three digit code to each character. For instance, 'A' is '065' and 'a' is '097'. The problem with ASCII is that it can only store 256 unique bytes (characters), which was not enough to encode characters from other languages. Thus, the **UTF-8** came into being. **UTF-8** is capable of encoding 1,112,064 symbols (yes, that includes emojii). For example, letter 'A' is encoded as 'U+0041'.
````

Normally, the first thing to check after importing the data is whether the data was imported correctly. This typically implies 1) checking the total number of intended variables (if we how many features/variables the original file had); 2) checking for correct type of variables. To check the number of records and variables in the dataset we can use the command **shape**, which returns a tuple (110,22), where the first number denotes the number of records and the second number denotes the number of variables. 

```python
print(df.shape) 
```

You can check out the types of data after reading the data file using a **dtypes** command on the data frame. 

```python
df.dtypes
```

### Calling Variables 

Variables (features) can be called using the following conventions

```python

df.variable_name

df['variable_name']
```

### Filtering and subsetting the data

We can subset the data either by label or index. 

Label-based filtering. 

```python
# select only records for California
ca_df = df.loc[df.state=="CA",] 

# select only records for CA and FL
ca_fl_df = df.loc[(df.state=="CA")|(df.state=="FL"),]

# if we only needed specific variables in the previous selections
ca_df2 = df.loc[df.state=="CA", ['first_var', 'second_var', 'third_var']]
```

Index-based filtering

```python
# select first 100 rows 
df.iloc[:100,]

# select first 100 rows and variables 2,3,4,5
df.iloc[:100, 2:5]
```

To select all records for specific variables use double square brackets like so:

```python
other_df = df[['first_var', 'second_var', 'third_var']]
```

## Introducing Data for the Lab

There are many wonderful resources online containing different types of data: **Kaggle, Google, Data.gov, Census, OpenStreetMap** and many more. Also, check out this wonderiful [Github repo](https://github.com/awesomedata/awesome-public-datasets). For this lab, we will be working with **fire perimeter data** from CalFire. The geodatabase dump can be donwloaded from this [page](https://frap.fire.ca.gov/mapping/gis-data/). Here is the description of the data from the CalFire website: 

> This is a multi-agency statewide database of fire history. For CAL FIRE, timber fires 10 acres or greater, brush fires 30 acres and greater, and grass fires 300 acres or greater are included. For the USFS, there is a 10 acre minimum for fires since 1950.

> This dataset contains wildfire history, prescribed burns and other fuel modification projects.

This data can also be viewed interactively on a [CA.gov](https://gis.data.ca.gov/datasets/CALFIRE-Forestry::california-fire-perimeters-all-1/explore?location=37.442493%2C-118.992700%2C7.54). 

The file is available as a ArcGIS personal geodatabase (file extension .gdb). Since we are reading it in GeoPandas some labels are not available, we will have to recode a variable 'CAUSE' manually, using the following values: 

```python
rec_vars = {
    1: 'Lightning', 
    2: 'Equipment Use', 
    3: 'Smoking', 
    4: 'Campfire', 
    5: 'Debris', 
    6: 'Railroad', 
    7: 'Arson', 
    8: 'Playing with fire', 
    9: 'Miscellaneous',
    10: 'Vehicle', 
    11: 'Powerline', 
    12: 'Firefighter Trainig', 
    13: 'Non-Firefighter Training', 
    14: 'Unknown'
}
```

## Modifying variables 

### Working with datetime 

Some variables denote date and time. These should normally be processed as 'datetime' variable, thus we convert them 

```python

df['my_date_variable'] = pd.to_datetime(df.my_date_variable)

```


## Calculating Measures of Location and Variation of Variables

Normally we are interested in assessing specific variables. In Pandas and GeoPandas the variables can be called using either square brackets or dot. See pseudo code below used to calculate averages for variables.  

```python
df.variable_name.mean()

df['variable_name'].mean()
```

Pandas implements convenience functions such as *mean(), median(), mode(), std()* to name a few. 


## Visualzing Data

### Setting up canvas for plotting 

Matplotlib remains a default backend for many utilitity functions in Pandas. Therefore, it is important to know how to change figure attributes. Here is a typical workflow for modifying canvas size, axis labels and a title. 

```python
fig, ax = plt.subplots(figsize=(10,10)) #figure size in inches 

df.variable_name.plot(kind='hist', ax=ax) # set axis to be axis we modified above

ax.set_title('My Title', fontsize=16)
ax.set_

```

### Histograms 

We discussed in class that data distributions can be visualized via histograms. Pandas makes this process extremely easy: 

```python
df.variable_name.plot(kind='hist')
```
For cases when the distributions are exponential, we could use the log-scale and density on y-axis to see more variation in our data:

```python
df.variable_name.plot(kind='hist', logy=True, density=True)
```

### Cartographic Mapping in GeoPandas 

Geometric information is typically stored as well-known text (WKT) format in GeoPandas dataframes. 

There are three basic types of GIS data: [multi]point, [multi]line and [multi]polygon. 

Creating maps in GeoPandas is easy and straight-forward. 

```python

gdf = gpd.read_file('path_to_file')

gdf.plot()
```

For polygons, we can use GeoPandas to generate choropleth map, by passing the name of a variable we want to use for thematic mapping, like so: 

```python
gdf.plot('my_variable')
```

## Lab Instructions

Describe and characterize wildfires in the assigned CA county. The assignment is available in the [Google Doc](https://docs.google.com/spreadsheets/d/1p2N6L2RgSAaXoJzuCG_Sb7gZIFuGuoED_3LGkzcxeKI/edit?usp=sharing). Find your assignment by a perm number. There are two students per county. It is up to you if you want to collaborate on your lab, but each of you should submit individual lab. Your code and plots should have different styling. There are still some counties that have not been assigned to anyone. If you want to work on some other county, feel free to put your PERM for any other county on the sign-up sheet.  

0. Create a geographic map of the county and fires
1. Calculate total number of fires in the county
2. Calculate average acreage within the county
3. Calculate average duration of wildfires in the county 
4. Plot the histogram of wildfire duration in the county 
5. Create a bar plot or a line plot with the average number of wildfires in each month 
6. Createa a time line of the total number of fires in the county (1910-2021) 
7. Plot average acreage of fires over time 
8. Download the average annual temperature for your data from NOAA
    1. Go to the [website](https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/county/time-series)
    2. Input the following: Parameter - Average Temperature, Time Scale - Annual, Start Year - 1950, End Year - 2021, State - California, County - County_Of_Your_Assignment
    3. Click 'Plot', scroll down and download data in csv format. 
    4. Upload the file to your Google drive and continue working. Alternatively right click on the excel logo above the rendered table to the right of the word 'Download' and click 'Copy link address'.  
9. Plot three subfigures (use any orientation you find useful): a) annual number of fires, b) total annual acreage of fires, c) average temperature (that you just downloaded). 
10. Edit your report with markdown headings and text where necessary. **MAKE SURE TO COMMENT AND INTERPRET EVERY PLOT**. 
11. Submit via GauchoSpace as *geog172_firstlastname_lab02.ipynb*. 
12. Optional 1: recode variable 'CAUSE' using a dictionary above and generate a bar chart with the most common cause of fires in CA. 
13. Optional 2: Plot the number of fires by cause over time. 

---