# Lab 3: Geovisualization 

````{caution}
This lab will provide an opportunity to practice choropleth mapping and point pattern analysis. Please start by opening [this Google Colab notebook](https://colab.research.google.com/drive/1bN83L4Q6pYDVuGEjhiN_3Jq7tFdb7E95?usp=sharing) and copying it to your Google Drive. 
````

````{important}
Please submit your lab by modifying the code in the [Lab 3 Google Colab template notebook](https://colab.research.google.com/drive/1x-g5TJsR5p3VcxZEMgL4WBwnjWWX4dHn?usp=sharing)
````

````{warning}
Please make sure to submit the lab by **October 17/18, 2022**, depending on your section.
````

## Lab Instructions

````{tip}

Use the **isin()** function to subset your data.

```python
my_region = ['California', 'Washington', 'Oregon']

my_states = df.loc[df['state_name'].isin(my_region)]
```
````

````{tip}

Use the datetime indexing. See more info [here](https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html)

```python

df = df.set_index('dt') 

new_df = df.loc['2014-01-01':'2015-01-01']

```

````

1. Find your area of interest: for last names starting with A-G (West), H-M (Midwest), N-S (Northeast), T-Z (South). Filter temperature data to include only the area of your interest. See [this link](https://www2.census.gov/geo/pdfs/maps-data/maps/reference/us_regdiv.pdf) for a list of state names. 
2. What is the average temperature in the area of your choice. (Use .mean() function). 
3. Group your data by year and plot average temperature in your assigned region as a time-series plot. 
4. Group your data by 'state' and caclulate average temperature for each state. Create a Figure with two subfigures. Add a choropleth map for each subfigure plotting average temperature per state using two different types of classification (choose any). 
5. Create two dataframes: **weather_1980_2000** and **weather_2000_2020** by filtering your data on year of observation. 
6. Group two newly created datasets by 'state' and caclulate average temperature for each state. Create a Figure with two subfires. Add a choropleth map generated from **weather_1980_2000** and **quantile** classification to the **left** subfigure. Add a choropleth map generated from **weather_2000_2020** and **quantile** classification to the **right** subfigure.
7. Create a Figure with two subfires. Add a choropleth map generated from **weather_1980_2000** and **mean standard deviation** classification to the **left** subfigure. Add a choropleth map generated from **weather_2000_2020** and **mean standard deviation** classification to the **right** subfigure. 
8. Comment on every plot using markdown to format your asnwer. 
9. Submit to GauchoSpace as an .ipynb notebook. 