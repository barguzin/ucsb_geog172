# Lab 7: Clustering the Blocks of Cincinnati

````{caution}
This lab will provide you with an opportunity to investigate the spatial crime patterns in Cincinnati. The second part will focus on working through your Data Interim Report (due November 13, 2022). Please start by opening [this Google Colab notebook](https://colab.research.google.com/drive/1F3rv7hDzxOVMPxy--D5DpYDUUMkRqe1A?usp=sharing) and copying it to your Google Drive. 
````

````{important}
Please submit your lab by modifying the notebook available at the link above!
````

````{warning}
Make sure to submit the lab by **November 14 / 15, 2022**, depending on the scheduled time of your section.
````

---

## Lab Instructions

1. Create a histogram for the following variables: 'BURGLARY', 'ASSAULT', 'THEFT' 
2. Create a choropleth map for the three variables 
3. Calculate Global Moran's I for for the three variables 
4. Create LISA (Local Moran's I) maps for the three variables 
5. Scale your data using robust_scale()
6. Run hierarchical clustering and plot your results on a geographic map 
7. Run hierarchical clustering with contiguity constraints and plot your results on a geographic map
8. Compare two geographic maps with clusters and explain which of the partitioning should be preferred and why. 
9. Submit to GauchoSpace. 