#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install geopandas shapely fiona pyproj rtree')


# In[2]:


import fiona 
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# ## Getting Data

# ### Fire Perimeters

# In[3]:


get_ipython().system(' wget https://frap.fire.ca.gov/media/3ufh3ajg/fire21_1.zip -O fire_perims.zip -nc')


# In[4]:


get_ipython().system(' unzip -n fire_perims.zip')


# In[5]:


get_ipython().system(' ls')


# In[6]:


gdb_file = 'fire21_1.gdb'

# Get all the layers from the .gdb file 
layers = fiona.listlayers(gdb_file)

for layer in layers:
  if layer == 'firep21_1': # there are 3 files in gdb, we only need one
    print(f'found file: {layer}')
    fires = gpd.read_file(gdb_file,layer=layer)


# In[7]:


fires.shape


# In[8]:


# there are 20k fires in the data set, lets randomly plot 1000 of them
fires.sample(1000).plot()


# In[9]:


fires.head()


# In[10]:


fires.dtypes


# In[11]:


# lets drop some variables that we will not be using 
print(fires.shape[1])
fires.drop(['STATE', 'COMMENTS', 'C_METHOD', 'OBJECTIVE'], axis=1, inplace=True)
print(fires.shape[1])


# In[12]:


# some mumbo jumbo with the dates 
fires['ALARM_DATE'] = fires.ALARM_DATE.str.slice(0,10)
fires['CONT_DATE'] = fires.CONT_DATE.str.slice(0,10)
fires


# In[13]:


# calculate the number of where date is missing
print('number of missing records for alarm date:', fires.ALARM_DATE.isnull().sum())
print('number of missing records for containment date:', fires.CONT_DATE.isnull().sum()) 


# In[14]:


# convert variables to datetime
fires['ALARM_DATE'] = pd.to_datetime(fires.ALARM_DATE, errors='coerce')
fires['CONT_DATE'] = pd.to_datetime(fires.CONT_DATE, errors='coerce')


# In[15]:


# calculate new variable duration 
#fires['dur_days'] = (fires.CONT_DATE - fires.ALARM_DATE).dt.days
fires['dur_days'] = (fires.CONT_DATE - fires.ALARM_DATE).astype('timedelta64[D]')

print('Values where alarm date is before containment date:', fires.loc[fires.dur_days<0].shape[0])


# In[16]:


fires.dur_days.isnull().sum()


# In[17]:


# drop where we have missing duration
print(fires.shape)
fires = fires.loc[fires.dur_days>0,]
print(fires.shape)


# ### California Counties

# In[18]:


ca_counties = gpd.read_file('https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/california-counties.geojson')
print(ca_counties.shape)
ca_counties.head()


# ### Projections! 

# In[19]:


# check 
print(fires.crs)
print(ca_counties.crs)


# In[20]:


# let's convert both of those to epsg:3857
# this will take a while - lots of calculations
fires = fires.to_crs('epsg:3857')
ca_counties = ca_counties.to_crs('epsg:3857')


# In[21]:


# plot fires on CA
fig,ax = plt.subplots(figsize=(10,10))

ca_counties.plot(ax=ax, facecolor='w', ec='k');
fires.sample(1000).plot(fc='r', ax=ax);

plt.title('1000 fires in CA');


# In[22]:


# THIS PART OF CODE IS OPENING A FILE VIA OGR - SKIP
# SOME LABELS ARE NOT READ BY GEOPANDAS 
# from osgeo import ogr

# driver = ogr.GetDriverByName("OpenFileGDB")
# ds = driver.Open("fire21_1.gdb", 0)
# f = ds.GetLayer("firep21_1")
# print(type(f))


# ## Measures of Location and Variation

# In[23]:


print('mean', fires.GIS_ACRES.mean())
print('median', fires.GIS_ACRES.median())
print('std', fires.GIS_ACRES.std())


# ## Visualizing Data

# ### Histograms

# In[24]:


fig, ax = plt.subplots(figsize=(12,4))

fires.GIS_ACRES.plot(density=True, logy=True, kind='hist', ax=ax);

# changing title and labels
ax.set_title('GIS acreage', fontsize=16);
ax.set_ylabel('log(density)');


# In[25]:


# create histograms for all numeric variables 
num_vars = fires.select_dtypes(include=np.number).columns.tolist()
num_vars


# In[26]:


fires[num_vars].hist(density=True, bins=30, figsize=(15,15));


# In[27]:


# we can also call histogram from matplotlib interface 
plt.hist(fires.GIS_ACRES);


# ## Asking interesting questions about data 
# 
# 1. What is the average size of wildfires in CA
# 2. Does average size of fires increase over time? 
# 3. How many fires do we have per year? 
# 4. Does the number of fires increase over time? 
# 5. How does average fire duration change? 

# In[28]:


# calculating average size of fires in CA 
fires.area.mean() # this is in square meters: 28,624,990


# In[29]:


# convert YEAR_ to int 
fires.YEAR_ = fires.YEAR_.astype(int)
fires.dtypes


# ### Grouping variables and summarizing

# In[30]:


# only use variables that you will be calculating on 
avg_acres_by_year = fires[['YEAR_', 'GIS_ACRES']].groupby('YEAR_')["GIS_ACRES"].mean().reset_index()

avg_acres_by_year.set_index('YEAR_', inplace=True)

avg_acres_by_year.plot(figsize=(12,4), title='Average area of wildfires\n in California', color='r', linestyle='dashed');


# In[31]:


# only use variables that you will be calculating on 
total_acres_by_year = fires[['YEAR_', 'GIS_ACRES']].groupby('YEAR_')["GIS_ACRES"].sum().reset_index()

total_acres_by_year.set_index('YEAR_', inplace=True)

total_acres_by_year.plot(figsize=(12,4), title='Total area of wildfires\n in California');


# In[32]:


print(fires.shape[0]) # number of records
print(fires.INC_NUM.count()) # number of fire_id 


# In[33]:


# calculate number of fires per year 
fires_per_year = fires[['YEAR_', 'INC_NUM']].groupby('YEAR_')["INC_NUM"].count().reset_index()

fires_per_year.set_index('YEAR_', inplace=True)

fires_per_year.plot(figsize=(12,4), title='Total number of wildfires\n in California');


# In[34]:


# zoom into specific years and change range on y-axis, remove legend label, change linestyle and marker style
fires_per_year.plot(figsize=(12,4), title='Total number of wildfires\n in California (1990-2020)', ylim=(0,350), xlim=(1989,2022), legend=False, color='k', marker='s', linestyle='-.');


# In[35]:


# total duration of fires per year
dur_per_year = fires[['YEAR_', 'dur_days']].groupby('YEAR_')["dur_days"].sum().reset_index()

dur_per_year.set_index('YEAR_', inplace=True)

dur_per_year.plot(figsize=(12,4), title='Total duration of wildfires in days\n in California');


# In[36]:


# get temperature 
ca_temps = pd.read_csv('https://www.ncei.noaa.gov/cag/statewide/time-series/4-tavg-12-12-1910-2022.csv?base_prd=true&begbaseyear=1901&endbaseyear=2000', skiprows=5, header=None, dtype={0:'str'}, nrows=100) # use this line if following lab instructions
#ca_temps = pd.read_csv('https://raw.githubusercontent.com/barguzin/ucsb_geog172/main/data/ca_avg_temps.csv', skiprows=5, header=None, dtype={0:'str'}, nrows=100)
print(ca_temps.shape)
ca_temps.head()


# In[37]:


ca_temps.columns = ['date_year', 'temp', 'anomaly']


# In[38]:


# prep year 
# convert to 
ca_temps.date_year = ca_temps.date_year.str.slice(0,4)

ca_temps.date_year = ca_temps.date_year.astype(int)

ca_temps.set_index('date_year', inplace=True)

ca_temps.tail()


# In[39]:


print(ca_temps.index.dtype)
print(dur_per_year.index.dtype)


# In[40]:


# plot total duration of wildfires and average temps on two subplots 

fig, (ax1,ax2) = plt.subplots(2,1,figsize=(12,8), sharex=True)

ca_temps.temp.plot(ax=ax1)
ax1.set_title('Average Temperature in CA')

dur_per_year.plot(ax=ax2)
ax2.set_title('Total duration of wildfires in CA')


# #### There is a problem with index alignment, we need to merge dataframes! 

# In[41]:


merged = ca_temps.join(dur_per_year)

fig, (ax1,ax2) = plt.subplots(2,1,figsize=(12,8), sharex=True)

merged.temp.plot(ax=ax1)
ax1.set_title('Average Temperature in CA')

merged.dur_days.plot(ax=ax2)
ax2.set_title('Total duration of wildfires in CA')


# ## Spatial join and some summary statistics 

# Let's imagine you work at the county GIS Department. You are tasked to describe the wildfire situation in that county. We need to be able to subset our data set by county. For that we need to run spatial join. 

# In[42]:


# join fires to counties 
print(fires.shape) # total number of fires before join 
sj = fires.sjoin(ca_counties[['name', 'geometry']], how='left') # only keep name and geometry other variables are not required
print(sj.shape) # total number of fires after join - notice some were duplicated because some fires stretch several couinties! 


# In[43]:


sj.head()


# In[44]:


fires_by_county = sj.groupby(['name'])['INC_NUM'].size().reset_index()

print(fires_by_county.shape)
fires_by_county.sort_values(by='INC_NUM', inplace=True)
print('top 10 counties with most fires')
fires_by_county[::-1][:10]


# In[45]:


print('bottom 10 counties with most fires')
fires_by_county[:10]


# #### Optional: plot number of fires in top 10 counties in California over time

# In[46]:


from google.colab import drive
import os

# if you want to save file to your google drive 
drive.mount('/content/drive/')


# In[60]:


# save to Drive
fires.to_file('/content/drive/MyDrive/geog172/fires.geojson')


# In[61]:


# we can also save it to the content of the GC session and then right click and download to local computer 
fires.to_file('fires.geojson')

