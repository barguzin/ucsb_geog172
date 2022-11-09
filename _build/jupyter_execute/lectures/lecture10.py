#!/usr/bin/env python
# coding: utf-8

# <h1> <center> GEOG 172: INTERMEDIATE GEOGRAPHICAL ANALYSIS </h1>
#     <h2> <center> Evgeny Noi </h2>
#         <h3> <center> Lecture 10: Global Spatial Autocorrelation </h3>

# # Global Spatial Autocorrrelation 
# 
# * Moran's $I$ is just one measure of spatial autocorrelation. There are multiple SA measures. 
# * Geary's $c$ 
# * Getis and Ord's $G$

# # Geary's $c$
# 
# $$
# C = \dfrac{(n-1)}
#           {2 \sum_i \sum_j w_{ij}} 
#     \dfrac{\sum_i \sum_j w_{ij} (y_i - y_{j})^2}
#           {\sum_i (y_i - \bar{y})^2}
# $$
# 
# where $n$ is the number of observations, $w_{ij}$ is the cell of binary matrix, and $\bar{y}$ is the sample mean. 
# 
# * When compared to Moran’s I, it is apparent both measures compare the relationship of $Y$ within each observation's local neighborhood. 
# * Moran’s I takes cross-products on the standardized values, Geary’s C uses differences on the values without any standardization.

# In[1]:


from libpysal.examples import load_example
import geopandas as gpd
guerry_loc = load_example('Guerry')
guerry = gpd.read_file(guerry_loc.get_path('guerry.geojson'))
print(guerry.shape)
guerry.head()


# In[2]:


from libpysal.weights.contiguity import Queen
import splot
from esda.moran import Moran

y = guerry['Crm_prp'].values
w = Queen.from_dataframe(guerry)
w.transform = 'r'

moran = Moran(y, w)
moran.I


# # Geary's C Permutation Inference 
# 
# * Inference is performed in a similar way as with Moran’s I. 
# * Through simulation we can draw an empirical distribution of the statistic under the null of spatial randomness, and then compare it with the statistic obtained when using the observed geographical distribution of the data. 
# * To access the pseudo p-value, calculated as in the Moran case, we can call p_sim:

# In[3]:


from esda.geary import Geary

geary = Geary(guerry["Crm_prp"], w)
print(f"The Geary's C is {geary.C}")
print(f"The simulated pseudo p-value is {geary.p_sim}")


# # Getis & Ord's G
# 
# * Global version of a family of statistics of spatial autocorrelation based on distance. Originally measure of 'concentration'
# * Originally conceived for points, thus we use distances in $W_{ij}$ (as opposed to binary)
# * It is designed for the study of positive variables with a natural origin
# 
# $$
# G(d) = \dfrac{ \sum_i \sum_j w_{ij}(d) \, y_i \, y_j }
#              { \sum_i \sum_j y_i \, y_j }
# $$
# 
# where $w_{ij}(d)$ is the binary weight assigned based on the distance bin. 

# # Getis & Ord's G 
# 
# * IMPORTANT! Best suited to test to what extent similar values (either high or low) tend to co-locate. $G$ is a statistic of positive spatial autocorrelation. Thus the statistic is not able to pick up cases of negative spatial autocorrelation.

# In[4]:


import pandas as pd
from pysal.lib import weights
pts = guerry.centroid
xys = pd.DataFrame({"X": pts.x, "Y": pts.y})
min_thr = weights.util.min_threshold_distance(xys)
min_thr


# # Getis & Ord's G
# 
# * Similarly, inference can also be carried out by relying on computational simulations that replicate several instances of spatial randomness using the values in the variable of interest, but shuffling their locations. In this case, the pseudo P-value computed suggests a clear departure from the hypothesis of no concentration.

# In[5]:


from esda.getisord import G

w_db = weights.DistanceBand.from_dataframe(guerry, min_thr)
gao = G(guerry["Crm_prp"], w_db)
print(
    "Getis & Ord G: %.3f | Pseudo P-value: %.3f" % (gao.G, gao.p_sim)
)


# # Bivariate Moran's
# 
# > The concept of bivariate spatial correlation is complex and often misinterpreted. It is typically considered to be the correlation between one variable and the spatial lag of another variable, as originally implemented in the precursor of GeoDa (e.g., as described in Anselin, Syabri, and Smirnov 2002). However, this does not take into account the inherent correlation between the two variables. More precisely, the bivariate spatial correlation is between $x_i$ and $\Sigma_j w_{ij}y_{j}$, but does not take into account the correlation between $x_i$ and $y_i$, i.e., between the two variables at the same location. <...> As a result, this statistic is often interpreted incorrectly, as it may overestimate the spatial aspect of the correlation that instead may be due mostly to the in-place correlation.
# 
# [source](https://geodacenter.github.io/workbook/5b_global_adv/lab5b.html)

# # Bivariate Moran's 
# 
# * In its initial conceptualization, as mentioned above, a bivariate Moran scatter plot extends the idea of a Moran scatter plot with a variable on the x-axis and its spatial lag on a y-axis to a bivariate context. The fundamental difference is that in the bivariate case the spatial lag pertains to a different variable. 
# 
# $$
# I_B = \frac{\sum_i (\sum_j w_{ij} y_j \times x_i)}{ \sum_i x_i^2},
# $$

# In[6]:


from esda.moran import Moran_BV

x = guerry['Litercy'].values
y = guerry['Crm_prp'].values

moran_bv = Moran_BV(y, x, w)
print(f"Bivariate Moran's I: {round(moran_bv.I, 3)}")


# In[7]:


import scipy 
import matplotlib.pyplot as plt
import numpy as np

plt.scatter(x,y)
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))

r, p = scipy.stats.pearsonr(x, y) 


# In[8]:


from splot.esda import plot_moran_bv_simulation, plot_moran_bv
import matplotlib.pyplot as plt

plot_moran_bv(moran_bv)
plt.show()


# # Bivariate Moran Facet Matrix

# In[9]:


from esda.moran import Moran_BV_matrix
from splot.esda import moran_facet
guerry_vars = guerry[['Crm_prs', 'Litercy', 'Donatns', 'Wealth']]
matrix = Moran_BV_matrix(guerry_vars, w)

moran_facet(matrix)
fig = plt.gcf()
fig.savefig('moran_facet.png')
plt.close()


# In[10]:


from IPython.display import Image
Image(filename='moran_facet.png') 


# # Even more types of autocorrelating one variable
# 
# * Differential Moran's $I$ 
# * Moran's Scatter plot for EB rates 
# * Spatial correlogram 
# 
# #### Check out [GeoDa documentation](https://geodacenter.github.io/documentation.html) for more details. 

# # GeoDa
# 
# <img src="https://geodacenter.github.io/images/nonspatial_clusters.png" width="500px">

# # GeoDa 
# 
# * Basic Mapping 
# * E(S)DA 
# * Spatial Weights 
# * Global Spatial Autocorrelation  
# * Local Spatial Autocorrelation 
# * Clustering / Regionalization
# * Dimension reduction 
