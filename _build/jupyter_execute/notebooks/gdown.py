#!/usr/bin/env python
# coding: utf-8

# # Instructions on how to upload your working file to Google Drive 
# 
# Once you have created the working file from your data report you can upload it to Google Drive using the procedure outlined below. The procedure utilizes the package *gdown* (so do not forget to 'pip install' it. I am using the guerry.zip file here, so all of the names can be tweaked according to whatever names you are using. 
# 
# Then follow the instructions below: 
# 
# 1. Zip your working file (guerry.zip)
# 2. Upload the file to your Google Drive. Right click the file > Share > General Access (Anyone with the link) > Viewer > Copy Link. 
# 3. Paste the link below for your convenience 
# ```
# https://drive.google.com/file/d/1DN2jTDbwdGhyIoR1YKkFZrEbode77_VF/view?usp=sharing
# ``` 
# 4. Copy only file id from the link above (all symbols bewteen 'file/d/' and '/view?'
# ```
# 1DN2jTDbwdGhyIoR1YKkFZrEbode77_VF
# ```
# 5. Paste copied file id into URL below, like so 
# ```
# https://drive.google.com/uc?id=1DN2jTDbwdGhyIoR1YKkFZrEbode77_VF
# ```
# 6. Use gdown to download the file to content/ directory by copying the link from '5'
# ```bash
# !gdown https://drive.google.com/uc?id=1DN2jTDbwdGhyIoR1YKkFZrEbode77_VF
# ```
# 7. Unzip the file using CLI. You will need to change the filename here with your own. 
# ```bash
# !unzip guerry.zip
# ```
# 8. Finally, read your data using pandas or geopandas depending on the type of your data.
# ```python
# import geopandas as gpd
# guerry = gpd.read_file('guerry/guerry.shp')
# guerry.plot()
# ```

# In[1]:


get_ipython().system('pip install gdown geopandas fiona rtree shapely pyproj')


# In[2]:


get_ipython().system(' gdown https://drive.google.com/uc?id=1DN2jTDbwdGhyIoR1YKkFZrEbode77_VF')

#https://drive.google.com/file/d/1DN2jTDbwdGhyIoR1YKkFZrEbode77_VF/view?usp=sharing


# In[3]:


get_ipython().system('ls')


# In[4]:


get_ipython().system('unzip guerry.zip')


# In[5]:


import geopandas as gpd
guerry = gpd.read_file('guerry/guerry.shp')
guerry.plot();

