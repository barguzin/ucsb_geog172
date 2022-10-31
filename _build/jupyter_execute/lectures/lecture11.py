#!/usr/bin/env python
# coding: utf-8

# <h1> <center> GEOG 172: INTERMEDIATE GEOGRAPHICAL ANALYSIS </h1>
#     <h2> <center> Evgeny Noi </h2>
#         <h3> <center> Lecture 11: Local Spatial Autocorrelation </h3>

# # Review 
# 
# * Correlation versus Autocorrelation
# * Global Spatial Autocorrelation (positive: HH/LL; negative: HL/LH). 
# * Moran's $I$, Geary's $C$ and Getis & Ord's $G$ 

# # Global versus Local Moran's $I$ 
# 
# $$
# I = \frac{\sum_i \sum_j w_{ij} z_iz_j}{\sum_i z_i^2},
# $$
# 
# $$
# I_i = \frac{\sum_j w_{ij} z_iz_j}{\sum_i z_i^2}.
# $$
# 
# where $z_i = x_i - \bar{x}$ (standardized value of the variable).

# # Local Moran's $I$ 
# 
# * Originally conceived to define local *spatial clusters* and *spatial outliers* 
#     * spatial clusters (high-high, low-low)
#     * spatial outliers (high-low, low-high)
# * The sum of the local statistics is proporational to the global Moran’s I, or, alternatively, that the global Moran’s I corresponds with the average of the local statistics
# * Significance is assessed via a permutation inference (!!! Multiple Comparisons)

# # Multiple Comparisons 
# 
# > Situation when statistical analysis involves multiple simultaneous statistical tests, each of which has a potential to produce a "discovery."). A stated confidence level generally applies only to each test considered individually, but often it is desirable to have a confidence level for the whole family of simultaneous tests. Failure to compensate for multiple comparisons can have important real-world consequences.
# 
# Take the case of $S = 100$ hypotheses being tested at the same time, all of them being true, with the size and level of each test exactly equal to $\alpha$. For $\alpha = 0.05$, one expects five true hypotheses to be rejected. Further, if all tests are mutually independent, then the probability that at least one true null hypothesis will be rejected is given by $1 − 0.95^{100} = 0.994$.

# # Multiple Comparisons and LISA 
# 
# * Traditional choice of 0.05 is likely to lead to many false positives, i.e., rejections of the null when in fact it holds.
# * Bonferonni bounds ($\alpha/n$) - very conservative (use to identify cluster centers) 
# * False Discovery Rate - less conservative 
#     * Sort $I_i$ p-values in the increasing order (variable $I$ denotes order) 
#     * $FDR = i \times \alpha/n$ (for first observation: $FDR = 1 \times 0.05/85 = 0.000118$ 
#     * Select all observations where $p_{i_{max}} \leq i \times \alpha / n$

# In[1]:


from libpysal.examples import load_example
import geopandas as gpd
guerry_loc = load_example('Guerry')
guerry = gpd.read_file(guerry_loc.get_path('guerry.geojson'))
print(guerry.shape)
guerry.head()


# In[2]:


from pysal.explore import esda  # Exploratory Spatial analytics
from pysal.lib import weights  # Spatial weights
from splot.esda import moran_scatterplot, lisa_cluster
from esda.moran import Moran, Moran_Local
from splot.esda import plot_local_autocorrelation
from splot.esda import plot_moran
import matplotlib.pyplot as plt


# In[7]:


y = guerry['Crm_prp'].values
w = weights.Queen.from_dataframe(guerry)
w.transform = 'r'

# calculate global
moran = Moran(y, w)
print("Moran's I:", moran.I)
print("Moran's p-val:", moran.p_sim)


# In[8]:


plot_moran(moran, zstandard=True, figsize=(10,4))
plt.show()


# <img src="https://geographicdata.science/book/_images/07_local_autocorrelation_21_0.png">

# In[9]:


# calculate Moran_Local and plot
moran_loc = Moran_Local(y, w)


# In[20]:


fig, ax = moran_scatterplot(moran_loc, p=0.05)
ax.set_xlabel('Crimes Againts Property')
ax.set_ylabel('Spatial Lag of Crimes Against Property')
plt.show()


# In[15]:


lisa_cluster(moran_loc, guerry, p=0.05, figsize = (5,5), aspect=1)
plt.show()


# In[18]:


# Calculate the p-value cut-off to control 
# for the false discovery rate (FDR) for multiple testing.
print(esda.fdr(moran_loc.p_sim, 0.05)) # Bonferonni
print(esda.fdr(moran_loc.p_sim, 0.1)) # FDF cut-off


# # Extensions to Local Moran's $I$ 
# 
# * Bivariate Local Moran's $I$ 
# * Differential Local Moran's $I$ 
# * Local Moran's $I$ with EB rates

# # Local Geary's $c$ 
# 
# $$
# LG_i = \sum_j w_{ij}(x_i - x_j)^2
# $$
# 
# While calculating Geary's c is possible in Python the plots are not yet automated. Use GeoDa for comparisons! 

# # Local Getis & Ord's $G$
# 
# $$
# G_i = \frac{\sum_{j \neq i} w_{ij} x_j}{\sum_{j \neq i} x_j} \quad \text{not including value at location i} 
# $$
# 
# $$
# G_i^* = \frac{\sum_j w_{ij} x_j}{\sum_j x_j}
# $$
# 
# In contrast to the Local Moran and Local Geary statistics, the Getis-Ord approach does not consider spatial outliers. Calculations similar to Local Geary's $C$

# # Interim Data Report
# 
# ## Class exercise 
# 
# > Work in pairs. Tell one another what data you are working in and what questions you hope to answer with the methods we've learned in class. 
# 
# ### Due: Sunday (Nov 6, 2022) 

# # Questions? 
