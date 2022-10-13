#!/usr/bin/env python
# coding: utf-8

# <h1> <center> GEOG 172: INTERMEDIATE GEOGRAPHICAL ANALYSIS </h1>
#     <h2> <center> Evgeny Noi </h2>
#         <h3> <center> Lecture 02: Basic Statistics </h3>

# # What is Statistics? 
# 
# * Statistics is the science concerned with developing and studying methods for collecting, analyzing, interpreting and presenting empirical data.
# * Two fundamental ideas in the field of statistics are uncertainty and variation. Probability is a mathematical language used to discuss uncertain events. 
# * Any measurement or data collection effort is subject to a number of sources of variation (i.e. if we repeat the same measurement, and re-run tests, then the answer would likely change).

# # Why do we need statistics? 
# 
# * Every year, the U.S. Census Bureau contacts over 3.5 million households across the country to participate in the American Community Survey. How can we assess the data? 
#     * Look at each survey individually... (cumbersome) 
#     * **Summarize and describe the data!** 

# # Different Types of Data 
# 
# * Continuous (interval, float, numeric) - *wind speed, time duration* 
# * Discrete (integer, count) - *count of tornadoes per state* 
# * Categorical (factors, nominal) - *state name* 
# * Binary (logical, boolean) - *true / false* 
# * Ordinal (explicit ordering) - *restaurant rating* 

# # Why Use Different Data Types 
# 
# * Optimize storage and computations 
# * Optimize predictions (stats procedures)

# # Rectangular Data
# 
# * Spreadsheet / database table. 
# * Data frame 
# * Feature (columns, independent variables) 
# * Outcome (dependent variable) 
# * Records (rows in a dataframe) 

# In[1]:


import configparser

# mumbo-jumbo to hide paths
config = configparser.ConfigParser()
config.read('config.ini')
data_path = config.get('my-config', 'data_path')


# In[2]:


import pandas as pd
import numpy as np
import geopandas as gpd
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
import cenpy
import contextily
from mpl_toolkits.axes_grid1 import make_axes_locatable


# In[3]:


df = pd.read_csv(data_path + 'metro-bike-share-trip-data.csv')
print(df.shape) 
df.head()


# In[4]:


df.columns


# In[5]:


df.dtypes


# # What is *int* and what is *64*? 
# 
# > Int64 is an immutable value type that represents signed integers with values that range from negative 9,223,372,036,854,775,808 through positive 9,223,372,036,854,775,807 (from Microsoft) 

# # Other data formats 
# 
# * Graphs 
# * Spatial data 
# * Spatio-temporal data

# # Summarizing and Describing the Data
# 
# ## Histograms 101
# 
# ### Can we describe duration of 132k bike trips? --> Easy! 

# In[6]:


df.Duration.value_counts()


# In[7]:


df.Duration.value_counts()[:10]


# In[8]:


df.Duration.value_counts()[:10].plot(kind='barh');


# In[9]:


df.Duration.plot(kind='hist'); # regular histogram


# <h2 style="color:blue"> <center> When we have wide range of values it is best to use log-scale to see variability </h2>

# In[10]:


df.Duration.plot(kind='hist', density=True, logy=True); # log-scale


# # How about some other data? 
# 
# ## Census API via Cenpy

# In[11]:


acs = cenpy.products.ACS()


# In[12]:


acs.tables


# In[13]:


acs.filter_tables('RACE', by='description')


# In[14]:


acs.filter_tables('HISPANIC', by='description')


# In[15]:


acs.filter_variables('B03002').head()


# In[16]:


hispanic = ['B03002_001', # full population 
            'B03002_002', # nonhispanic
            'B03002_012' # hispanic 
           ]


# In[17]:


sb = acs.from_place('Santa Barbara, CA', variables=hispanic)


# In[18]:


sb.head()


# In[19]:


sb_basemap, sb_extent = contextily.bounds2img(*sb.total_bounds, zoom=12, 
                                                        source=contextily.providers.Stamen.TonerLite)


# In[20]:


fig, ax = plt.subplots(1,1, figsize=(12,5))
ax.imshow(sb_basemap, extent=sb_extent, interpolation='sinc')
sb['pct_hispanic'] = sb.eval('B03002_012E / B03002_001E')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
sb.plot('pct_hispanic', cmap='Blues', ax = ax, alpha=.5, legend=True, cax=cax, 
       legend_kwds={'label': "Pct Hispanic"})
                    #, 'orientation': "horizontal"})
_ = ax.axis("off")


# In[21]:


print(sb.shape)
sb.pct_hispanic.plot(kind='hist');


# In[22]:


lv = acs.from_place('Las Vegas, NV', variables=hispanic)


# In[23]:


sa = acs.from_place('Sacramento, CA', variables=hispanic)


# In[24]:


lv['pct_hispanic'] = lv.eval('B03002_012E / B03002_001E')
sa['pct_hispanic'] = sa.eval('B03002_012E / B03002_001E')

fig, ax = plt.subplots(1,2, sharey=True, figsize=(12,4));
lv.pct_hispanic.plot(kind='hist',ax=ax[0], alpha=.5, color='red');
lv.pct_hispanic.plot(kind='kde',ax=ax[0], alpha=1, color='red');
ax[0].set_title('% Hispanic - Las Vegas');
sa.pct_hispanic.plot(kind='hist',ax=ax[1], alpha=.5, color='blue');
sa.pct_hispanic.plot(kind='kde',ax=ax[1], alpha=1, color='blue');
ax[1].set_title('% Hispanic - Sacramento');


# # Measures of center and spread 
# 
# * Center == Location 
# * Spread == Variation

# # Estimates of Location 
# 
# * Mean/Average - The sum of all values divided by the number of values.
# * Weighted Mean - The sum of all values times a weight divided by the sum of the weights.
# * Median - The value such that one-half of the data lies above and below.
# * Trimmed/Truncated mean - The average of all values after dropping a fixed number of extreme values.

# # Mean 
# 
# $$
# \text{Mean} = \bar{x} = \frac{\sum_{i=1}^{n}x_i}{n}
# $$
# 
# where $n$ is the size of the sample. 

# In[25]:


x = [1,11,3,12,5]
mean = np.sum(x)/5
print(mean)


# # Median 
# 
# * Mean is very sensitive to outliers (think about income and Bill Gates) 
# * Median is a middle of a sorted list of data (more robust to outliers) 
# * Trimmed mean is also robust to outliers (trip top 10% of high-income households) 

# In[26]:


print(np.median(x))


# # Estimates of Variability
# 
# * Deviation (errors, residuals) - The difference between the observed values and the estimate of location.
# * Variance - The sum of squared deviations from the mean divided by n – 1 where n is the number of data values.
# * Standard deviation - The squared root of variance 
# * Mean Absolute Deviation - The mean of the absolute value of the deviations from the mean.
# * Median <...> 

# * Range - The difference between the largest and the smallest value in a data set.
# * Order statistics (ranks) - Metrics based on the data values sorted from smallest to biggest.
# * Percentile - The value such that P percent of the values take on this value or less and (100–P) percent take on this value or more.
# * IQR - The difference between the 75th percentile and the 25th percentile.

# $$
# \text{Variance} = s^2 = \frac{\sum(x - \bar{x})^2}{n-1}
# $$
# 
# $$
# \text{Standard Deviation} = s = \sqrt{\text{Variance}}
# $$

# > the Pth percentile is a value such that at least P percent of the values take on this value or less and at least (100 – P)
# percent of the values take on this value or more. For example, to find the 80th percentile, sort the data. Then, starting with the smallest value, proceed 80 percent of the way to the largest value. Note that the median is the same thing as the 50th
# percentile. (Practical Statistics for Data Scientists, p. 41) 

# # Locations and Variability May Vary 

# In[27]:


mu, sigma = 0, 1 # mean and standard deviation
mu2, sigma2 = 1, 2

s = np.random.normal(mu, sigma, 1000)
s2 = np.random.normal(mu2, sigma2, 1000)


# In[28]:


fig, ax = plt.subplots(1,2, figsize=(12,4))
count, bins, ignored = ax[0].hist(s, 30, density=True, color='r', alpha=.2)
count2, bins2, ignored2 = ax[1].hist(s2, 30, density=True, color='b', alpha=.2)
ax[0].plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r');
ax[1].plot(bins2, 1/(sigma2 * np.sqrt(2 * np.pi)) *
               np.exp( - (bins2 - mu2)**2 / (2 * sigma2**2) ),
         linewidth=2, color='b');


# In[29]:


fig, ax = plt.subplots(1,1, figsize=(12,4))
count, bins, ignored = ax.hist(s, 30, density=True, color='r', alpha=.2)
count2, bins2, ignored2 = ax.hist(s2, 30, density=True, color='b', alpha=.2)
ax.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')
ax.plot(bins2, 1/(sigma2 * np.sqrt(2 * np.pi)) *
               np.exp( - (bins2 - mu2)**2 / (2 * sigma2**2) ),
         linewidth=2, color='b')


# # Skewness and Kurtosis
# 
# * Skewness is a measure of symmetry, or more precisely, the lack of symmetry. A distribution, or data set, is symmetric if it looks the same to the left and right of the center point. 
# * Kurtosis is a measure of whether the data are heavy-tailed or light-tailed relative to a normal distribution. That is, data sets with high kurtosis tend to have heavy tails, or outliers.

# <img src="https://www.researchgate.net/publication/340417580/figure/fig2/AS:876314567380995@1585941082906/Skewness-and-kurtosis.jpg" width="500px">

# # Many different Types of Distribution
# 
# * Normal 
# * Poisson 
# * Gamma 
# * Beta
# * Uniform
# * Chi-Square
# * many more ... 

# # Normal Distribution
# 
# * We already established measures of location ($\mu$) and variability ($\sigma$). 
# * If we parametrize the distribution with two variables, such that it's density: 
# 
# $$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x - \mu}{\sigma}\right)^2}$$

# <img src="https://upload.wikimedia.org/wikipedia/commons/8/8c/Standard_deviation_diagram.svg">

# # Normal Distribution
# 
# * 68.2% of observations fall within one standard deviation from the mean
# * 95.4% (+27.2%) of observations fall within two standard deviations from the mean   
# * 99.6% (+4.2%) of observations fall within three standard deviations from the mean
# * Unimodal! 

# # Exercise: calculating SD
# 
# > Input: [1,9,7,4,2,3]
# 
# 1. Calculate mean
# 2. Calculate deviation from each point to the mean and square it 
# 3. Calculate variance (sum deviations over sample size)
# 4. Take square root of variance 

# In[30]:


np.std([1,9,7,4,2,3])


# # Examples of Normal Distribution
# 
# * Height 
# * Weight 
# * Dice outcomes 
# * IQ
# * Shoe size 
# * many more ... 

# # Central Limit Theory 
# 
# * One reason why normal (Gaussian) distribution is so popular 
# * Imagine we have a sample of size 1000 (let's assume that it's height in cm) 
# * If we randomly draw samples of size 10 with replacement and take the mean of those samples, the distribution of means will also have a normal distribution. 
# * This also works for sampling from non-normal distributions

# In[31]:


wh = pd.read_csv('weight-height.xls')
print(wh.shape) 
wh.head()


# In[32]:


i = 100
samples = []

while i>0:
    x = np.random.choice(wh.Height, 35)
    samples.append(np.mean(x))
    i = i - 1

fig, ax = plt.subplots(figsize=(10,4))
ax.hist(wh.Height, density=True, alpha=.5, label='Height')
ax.hist(samples, density=True, alpha=.5, label='Mean of Height');
ax.legend();


# # Questions? 
# 
# ## For reviewing 
# 
# * Think Stats (Chapter 2) + Practical Statistics for Data Scientists (Chapter 1) 
