# Interim Report

These guidelines will help you apply the methods learned in lecture and labs onto your geographical analysis project with the data that you collected for your *Data Report*. The key to successful interim report is being tedious. State your hypotheses, use visualizations to derive interesting insights from the data. Support your visual insights using statistical tests we have covered. 

````{warning}
**The Data Report is due 11/06/2022 by 23.59.** 
````

---

## Requirements 

Treat this report as a standalone document to that you prepare for your supervisor at potential job. You are given the data set and your task is to tell interesting story using graphs, plots, maps, followed by statistical hypotheses to support your initial findings derived from visualizations. 

1. At least two scatter plots or bar charts or pie charts or time series plot (or any other type of a plot)
2. At least one summary table (crosstab table). This can be the table that you would use for plotting. For instance aggregate statistic (mean, sum, etc.) per state that you will be using to create a choropleth map. 
3. At least one correlation coefficient 
4. At least one one sample t/z-test
5. At least one two sample test 
6. At least one ANOVA test 
7. At least one Global Moran's $I$ 
8. At least one Local Moran's $I$. This can be the same variable as in (7). 
9. At least one Moran scatter plot 
10. At least one density plot with simulated $p$-value 
11. At least one choropleth map with clusters (hotspots/coldspots) and outliers

````{important}
Make sure to follow the hypothesis steps outlined in the Hypothesis testing slides: **state hypotheses, set your confidence interval, run the test, get p-value, interpret your p-value by either failing to reject or rejecting null**.
````

--- 

## Project Instructions 

1. Create a new Google Colab notebook 
2. Import the data from your interim report into the notebook 
````{tip}
Use the code from the recently announced example [notebook](notebooks/../../notebooks/gdown.ipynb) and gdown.
````
3. Break your report into three parts: Visual Exploration, Statistical Tests, Geographic Analysis. Use level-1 heading for titles of these parts (i.e. \# Part name). 
4. Add content to the report, by including all of the items found in the 'Project Requirements' section. 
5. Provide hypotheses and interpretations for corresponding parts. 
6. Weave the plots and interpretations into a coherent document. The data story should be easy to follow. 

````{tip}
Examples of well-written notebooks for exploratory data analysis: 

* [Housing prices on Kaggle](https://www.kaggle.com/code/ekami66/detailed-exploratory-data-analysis-with-python/notebook)
* [Towards Data Science Blog Page](https://towardsdatascience.com/my-6-part-powerful-eda-template-that-speaks-of-ultimate-skill-6bdde3c91431). If the page does not open or if you have exceeded the amount of free articles on Medium blog platform, right-click and open in the incognito window. 
* [Collection of top 10 Kaggle notebooks](https://analyticsindiamag.com/top-ten-kaggle-notebooks-for-data-science-enthusiasts-in-2021/)
````
7. Submit the following to GauchoSpace. 
   1. Data (as either a csv or a geojson). This step is not required if you used the 'gdown' approach described above. 
   2. Notebook in ipynb format.
   3. Notebook as exported pdf. 

````{caution}
Your submitted notebook must run and/or have all of the cell output rendered, so that it could be graded. 
````