#!/usr/bin/env python
# coding: utf-8

# <h1> <center> GEOG 172: INTERMEDIATE GEOGRAPHICAL ANALYSIS </h1>
#     <h2> <center> Evgeny Noi </h2>
#         <h3> <center> Lecture 13: Clustering and Regionalization </h3>

# # Review 
# 
# * Many of the techniques we covered had to do with either one variable (autocorrelation) or pairs of variables (correlation). Today from *univariate* to *multivariate*.

# # Clustering 
# 
# * Part of unsupervised learning. 
#     * Latent patterns in our data 
#     * NO LABELS! 
# * Groups are not pre-determined (classification)
# * DBSCAN - example of density-based clustering

# # Classical clustering tasks
# 
# * Online shoppers (segmentation) 
# * Spotify users who like similar music
# * Hospital Patients

# # Clustering 
# 
# > **Partition** the observations into groups, where each observation is **similar** to another member of the group.
# 
# * Partitioning (pre-define number of clusters, pre-defined number of members in a group, pre-define radius) 
# * Similarity can be measured differently (attribute space - multidimensional, we can also add geography or spaital similarity to the clustering)

# # Regionalization 
# 
# > We can think of clusters as regions (similar groups of observations that have common attributes and spatial proximity) 

# # Types of Clustering Algorithms 
# 
# * Connectivity-based (hierarchical) 
# * Centroid-based (k-means, k-medoid, c-means, etc.) 
# * Distribution-based (clusters are groups most likely belonging to the same distribution) 
# * Density-based (DBSCAN) 
# * Grid-based 

# In[1]:


import pandas as pd
import geopandas as gpd
from esda.moran import Moran
from libpysal.weights import Queen, KNN
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg


# # Using PySAL example data

# In[2]:


from libpysal.examples import load_example, available

#[_ for _ in available().Name]

cin = load_example('Cincinnati')
cin.get_file_list()


# In[3]:


cin_df = gpd.read_file(cin.get_path('cincinnati.shp'))
print(cin_df.shape)
cin_df.head()


# In[4]:


cin_df[['WHITE', 'AP_WHITE']].describe()


# In[5]:


cin_df[['OCCHU_OWNE', 'OCCHU_RENT']].describe()


# In[6]:


cin_df.plot()


# In[7]:


cin_df.info()


# In[8]:


# consider only specific variables for clustering 
# see variable description at GEODA lab 
# https://geodacenter.github.io/data-and-lab/walnut_hills/
predictors = ['POPULATION', 'MEDIAN_AGE', 'AGE_65', 'WHITE', 'BLACK', 'ASIAN', 
          'NH_WHITE', 'HOUSEHOLDS', 'AVG_HHSIZE', 'HU_VACANT', 'OCCHU_OWNE', 'OCCHU_RENT']

crime_vars = ['BURGLARY', 'ASSAULT', 'THEFT']

# more examples with plotting available from 
# https://geographicdata.science/book/notebooks/10_clustering_and_regionalization.html


# In[9]:


f, axs = plt.subplots(nrows=3, ncols=4, figsize=(16, 12))
# Make the axes accessible with single indexing
axs = axs.flatten()
# Start a loop over all the variables of interest
for i, col in enumerate(predictors):
    # select the axis where the map will go
    ax = axs[i]
    # Plot the map
    cin_df.plot(
        column=col,
        ax=ax,
        scheme="Quantiles",
        linewidth=0,
        cmap="BuPu",
    )
    # Remove axis clutter
    ax.set_axis_off()
    # Set the axis title to the name of variable being plotted
    ax.set_title(col)

f.savefig('crime_predictors.png')
plt.close()


# In[10]:


from IPython.display import Image
Image(filename='crime_predictors.png') 


# In[11]:


f, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
# Make the axes accessible with single indexing
axs = axs.flatten()
# Start a loop over all the variables of interest
for i, col in enumerate(crime_vars):
    # select the axis where the map will go
    ax = axs[i]
    # Plot the map
    cin_df.plot(
        column=col,
        ax=ax,
        scheme="Quantiles",
        linewidth=0,
        cmap="BuPu",
    )
    # Remove axis clutter
    ax.set_axis_off()
    # Set the axis title to the name of variable being plotted
    ax.set_title(col)

f.savefig('crime_types.png')
plt.close()


# In[12]:


from IPython.display import Image
Image(filename='crime_types.png') 


# In[13]:


w = Queen.from_dataframe(cin_df)

# Calculate Moran's I for each variable
mi_results = [
    Moran(cin_df[variable], w) for variable in predictors
]
# Structure results as a list of tuples
mi_results = [
    (variable, res.I, res.p_sim)
    for variable, res in zip(predictors, mi_results)
]
# Display on table
table = pd.DataFrame(
    mi_results, columns=["Variable", "Moran's I", "P-value"]
).set_index("Variable")
table


# > Each of the variables displays significant positive spatial autocorrelation, suggesting clear spatial structure in the socioeconomic geography of San Diego. This means it is likely the clusters we find will have a non random spatial distribution. 

# In[14]:


xy_vars = predictors + ['BURGLARY']

corr_df = pg.pairwise_corr(cin_df[xy_vars], method='pearson')
corr_df.loc[corr_df.Y == 'BURGLARY'].sort_values(by='r', ascending=False)


# # Similarity 
# 
# * Absolute values will drown variables which have higher range of values (0-1 Vs 0-1000)
# * Standardize values by mean and stdev 
# * **For non-normal, skewed and bimodal distributions robust scaling may be required!**
# * Use **sklearn** package 

# # Standardizing Variables 
# 
# * **scale()**: $z = \frac{x_i - \bar{x}}{\sigma_x}$
# * **robust_scale()**: $z = \frac{x_i - \tilde{x}}{\lceil x \rceil_{75} - \lceil x \rceil_{25}}$ (median and IQR)
# * **minmax_scale()**: $z = \frac{x - min(x)}{max(x-min(x))}$ 

# In[15]:


from sklearn.preprocessing import robust_scale

db_scaled = robust_scale(cin_df[predictors])


# # K-means Clustering 
# 
# * Pre-specified number of clusters (groups) 
# * Each observation is closer to the mean of its own group than it is to the mean of any other group 

# # K-means Clustering Algorithm 
# 
# 1. Assign all observations to one of the $k$ labels 
# 2. Calculate multivariate mean over all covaraites for each cluster 
# 3. Reassign observations to the cluster with closest mean 
# 4. Repeat and update until there are no more changes

# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/K-means_convergence.gif/617px-K-means_convergence.gif">

# In[16]:


# Initialise KMeans instance
from sklearn.cluster import KMeans
# Initialise KMeans instance
kmeans = KMeans(n_clusters=5)
# Set the seed for reproducibility
np.random.seed(1234)
# Run K-Means algorithm
k5cls = kmeans.fit(db_scaled)
# Print first five labels
k5cls.labels_[:5]


# In[17]:


# Assign labels into a column
cin_df["k5cls"] = k5cls.labels_
# Setup figure and ax
f, ax = plt.subplots(1, figsize=(9, 9))
# Plot unique values choropleth including
# a legend and with no boundary lines
cin_df.plot(
    column="k5cls", categorical=True, legend=True, linewidth=0, ax=ax
)
# Remove axis
ax.set_axis_off()
f.savefig('kmeans.png')
plt.close()


# In[18]:


Image(filename='kmeans.png') 


# # Characterizing Clusters 
# 
# * Very imbalanced partitioning 
# * Some contiguity is noticeable, but the arrangement is patchy 

# In[19]:


# Group data table by cluster label and count observations
k5sizes = cin_df.groupby("k5cls").size()
k5sizes


# In[20]:


# Dissolve areas by Cluster, aggregate by summing,
# and keep column for area
areas = cin_df.dissolve(by="k5cls", aggfunc="sum")['geometry'].area
areas


# In[21]:


# Group table by cluster label, keep the variables used
# for clustering, and obtain their mean
k5means = cin_df.groupby("k5cls")[predictors].mean()
# Transpose the table and print it rounding each value
# to three decimals
k5means.T.round(3)


# In[22]:


# Index db on cluster ID
tidy_db = cin_df.set_index("k5cls")
# Keep only variables used for clustering
tidy_db = tidy_db[predictors]
# Stack column names into a column, obtaining
# a "long" version of the dataset
tidy_db = tidy_db.stack()
# Take indices into proper columns
tidy_db = tidy_db.reset_index()
# Rename column names
tidy_db = tidy_db.rename(
    columns={"level_1": "Attribute", 0: "Values"}
)
# Check out result
tidy_db.head()


# In[23]:


# Scale fonts to make them more readable
sns.set(font_scale=1.5)
# Setup the facets
facets = sns.FacetGrid(
    data=tidy_db,
    col="Attribute",
    hue="k5cls",
    sharey=False,
    sharex=False,
    aspect=2,
    col_wrap=3
)
# Build the plot from `sns.kdeplot`
_ = facets.map(sns.kdeplot, "Values", fill=True, warn_singular=False).add_legend()


# # Hierarchical Clustering 
# 
# * Agglomerative Hierarchical Clustering (AHC)
# * Hierarchy of solutions (starts at singletons) and assign all observations into same cluster 
# * We are looking for in-between clustering solutions 

# # AHC Algorithm 
# 
# 1. Every observation is its own cluster 
# 2. Find two closest observations based on distance metric (Euclidean Distance) 
# 3. Join the closest into a new cluster 
# 4. Repeat 2 and 3 until reaching the desired degree of aggregation

# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/ad/Hierarchical_clustering_simple_diagram.svg/1280px-Hierarchical_clustering_simple_diagram.svg.png" width='500px'>

# In[24]:


from sklearn.cluster import AgglomerativeClustering

# Set seed for reproducibility
np.random.seed(0)
# Iniciate the algorithm
model = AgglomerativeClustering(linkage="ward", n_clusters=5)
# Run clustering
model.fit(db_scaled)
# Assign labels to main data table
cin_df["ward5"] = model.labels_


# In[25]:


ward5sizes = cin_df.groupby("ward5").size()
ward5sizes


# In[26]:


ward5means = cin_df.groupby("ward5")[predictors].mean()
ward5means.T.round(3)


# In[27]:


# Index db on cluster ID
tidy_db = cin_df.set_index("ward5")
# Keep only variables used for clustering
tidy_db = tidy_db[predictors]
# Stack column names into a column, obtaining
# a "long" version of the dataset
tidy_db = tidy_db.stack()
# Take indices into proper columns
tidy_db = tidy_db.reset_index()
# Rename column names
tidy_db = tidy_db.rename(
    columns={"level_1": "Attribute", 0: "Values"}
)
# Check out result
tidy_db.head()


# In[28]:


# Setup the facets
facets = sns.FacetGrid(
    data=tidy_db,
    col="Attribute",
    hue="ward5",
    sharey=False,
    sharex=False,
    aspect=2,
    col_wrap=3,
)
# Build the plot as a `sns.kdeplot`
facets.map(sns.kdeplot, "Values", fill=True, warn_singular=False).add_legend()


# In[29]:


cin_df["ward5"] = model.labels_
# Setup figure and ax
f, axs = plt.subplots(1, 2, figsize=(12, 6))

### K-Means ###
ax = axs[0]
# Plot unique values choropleth including
# a legend and with no boundary lines
cin_df.plot(
    column="ward5",
    categorical=True,
    cmap="Set2",
    legend=True,
    linewidth=0,
    ax=ax,
)
# Remove axis
ax.set_axis_off()
# Add title
ax.set_title("K-Means solution ($k=5$)")

### AHC ###
ax = axs[1]
# Plot unique values choropleth including
# a legend and with no boundary lines
cin_df.plot(
    column="k5cls",
    categorical=True,
    cmap="Set3",
    legend=True,
    linewidth=0,
    ax=ax,
)
# Remove axis
ax.set_axis_off()
# Add title
ax.set_title("AHC solution ($k=5$)")

# Display the map
f.savefig('two_clust.png')
plt.close()


# In[30]:


Image(filename='two_clust.png') 


# # Regionalization 
# 
# ## Spatially Constrained Hierarchical Clustering

# # Why Regionalization 
# 
# * Clustering helps us investigate the structure of our data (spatial contiguity is not always a requirement) 
# * We impose spatial constraint on clusters (geographically coherent areas + coherent data profiles) 
# * Counties within states (administrative principles), clusters within out data (statistical similarity) 
# * Spatial weights matrix can be used as a measure of spatial similarity 

# In[31]:


# Set the seed for reproducibility
np.random.seed(123456)
# Specify cluster model with spatial constraint
regi = AgglomerativeClustering(
    linkage="ward", connectivity=w.sparse, n_clusters=5
)
# Fit algorithm to the data
regi.fit(db_scaled)


# In[32]:


cin_df["ward5wq"] = regi.labels_
# Setup figure and ax
f, ax = plt.subplots(1, figsize=(9, 9))
# Plot unique values choropleth including a legend and with no boundary lines
cin_df.plot(
    column="ward5wq",
    categorical=True,
    legend=True,
    linewidth=0,
    ax=ax,
)
# Remove axis
ax.set_axis_off()
# Display the map
plt.show()


# # Why are we getting imbalanced clusters? 
# 
# * Data is not normal 
# * Methods are not robust to outliers 

# In[33]:


cin_df[predictors].plot.hist(subplots=True, legend=True, layout=(4,3), figsize=(16,12), sharex=False);


# # K-Medoids Clustering (PAM)
# 
# * Partitioning around medoid (instead of 'mean', select most central actual observation). 
# * The medoid of a cluster is defined as the object in the cluster whose average dissimilarity to all the objects in the cluster is minimal, that is, it is a most centrally located point in the cluster. 
# * $k$-medoids minimizes a sum of pairwise dissimilarities instead of a sum of squared Euclidean distances (depends on implementation), it is more robust to noise and outliers than k-means  

# In[34]:


from sklearn_extra.cluster import KMedoids

kmedoids = KMedoids(n_clusters=5, random_state=0)

medo = kmedoids.fit(db_scaled)


# In[35]:


cin_df["medo_k5"] = medo.labels_
# Setup figure and ax
f, ax = plt.subplots(1, figsize=(9, 9))
# Plot unique values choropleth including a legend and with no boundary lines
cin_df.plot(
    column="medo_k5",
    categorical=True,
    legend=True,
    linewidth=0,
    ax=ax,
)
# Remove axis
ax.set_axis_off()
# Display the map
plt.show()


# <img src="https://scikit-learn.org/stable/_images/sphx_glr_plot_cluster_comparison_001.png">

# In[36]:


# add xy to clustering
cin_df2 = cin_df.copy()

# calculate centroid locs
cin_df2['lng'] = cin_df2.centroid.x
cin_df2['lat'] = cin_df2.centroid.y

# pct vars 
cin_df2['pct_elder']  = cin_df2.AGE_65 / cin_df2.POPULATION
cin_df2['pct_white']  = cin_df2.WHITE / cin_df2.POPULATION
cin_df2['pct_black']  = cin_df2.BLACK / cin_df2.POPULATION
cin_df2['pct_own'] = cin_df2.OCCHU_OWNE / cin_df2.HSNG_UNITS
cin_df2['pct_rent'] = cin_df2.OCCHU_RENT / cin_df2.HSNG_UNITS
cin_df2['pct_vac'] = cin_df2.HU_VACANT / cin_df2.HSNG_UNITS

cin_df2 = cin_df2.fillna(0)

new_predictors = ['POPULATION', 'MEDIAN_AGE', 'lng', 'lat', 'pct_elder', 'pct_white', 'pct_black', 
                 'pct_own', 'pct_rent', 'pct_vac']


# In[37]:


db_scaled2 = robust_scale(cin_df2[new_predictors])


# In[40]:


# Set the seed for reproducibility
np.random.seed(123456)
# Specify cluster model with spatial constraint
regi2 = AgglomerativeClustering(
    linkage="ward", 
    #connectivity=w.sparse, 
    n_clusters=5
)
# Fit algorithm to the data
regi2.fit(db_scaled2)


# In[41]:


cin_df2["ward5wq"] = regi2.labels_
# Setup figure and ax
f, ax = plt.subplots(1, figsize=(9, 9))
# Plot unique values choropleth including a legend and with no boundary lines
cin_df2.plot(
    column="ward5wq",
    categorical=True,
    legend=True,
    linewidth=0,
    ax=ax,
)
# Remove axis
ax.set_axis_off()
# Display the map
plt.show()


# In[42]:


# Set the seed for reproducibility
np.random.seed(123456)
# Specify cluster model with spatial constraint
regi2 = AgglomerativeClustering(
    linkage="ward", 
    connectivity=w.sparse, 
    n_clusters=5
)
# Fit algorithm to the data
regi2.fit(db_scaled2)


# In[43]:


cin_df2["ward5wq"] = regi2.labels_
# Setup figure and ax
f, ax = plt.subplots(1, figsize=(9, 9))
# Plot unique values choropleth including a legend and with no boundary lines
cin_df2.plot(
    column="ward5wq",
    categorical=True,
    legend=True,
    linewidth=0,
    ax=ax,
)
# Remove axis
ax.set_axis_off()
# Display the map
plt.show()


# # Questions
# 
# #### Learn more: https://geographicdata.science/book/notebooks/10_clustering_and_regionalization.html
