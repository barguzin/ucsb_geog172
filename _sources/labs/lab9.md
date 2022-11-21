# Lab 9: Regression (optional)

````{caution}
This labb will provide you with an opportunity to practice regression modeling in Python. We will be working with Guerry moral statistics data, that we already came across when studying LISA. Please start by opening [this Google Colab notebook](https://colab.research.google.com/drive/1INLpY-US5YN5BzM7r9bdtzjDWVAM64V8?usp=sharing) and copying it to your Google Drive. 
````

````{important}
Please submit the **.ipynb** notebook by modifying the code from the Google Colab notebook above.
````

````{warning}
Make sure to submit the lab by **November 28 / 29, 2022**, depending on the scheduled time of your section.
````

--- 

# Lab Instructions 

1. Check the normality of your outcome variable *MEDV* (hint: use a histogram). Create a correlation matrix for all of the variables.  
2. Fit the linear regression to the data using the following predictors: *CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT* to model *MEDV*. We will refer to this model as **m1**.  
3. Report and interpret your R-squared and Adjusted R-squared. Comment on the coefficients and denote which ones are significant and insignificant (given their p-values)
4. Now fit only predictors that only had significant p-values (call this model **m2**). 
5. Report and interpret your R-squared and Adjusted R-squared. Comment on the coefficients and denote which ones are significant and insignificant (given their p-values).
6. Compare **m1** to **m2**. Which factors appear to be most significant for *MEDV*? 
7. Calculate and report Global Moran's I for **m2** residuals. 
8. Plot **m2** residuals versus the lag of residuals. 
9. Calculate Local Moran's I and plot it on the map. Is there any spatial structure to **m2** residuals? 
10. Fit a fixed effect model **m3**. Report which neighborhood has the highest and lowest coefficient. Interpret. Report and interpret your R-squared and Adjusted R-squared. Comment on the coefficients and denote which ones are significant and insignificant (given their p-values).
11. Run the spatial error model **m4**. Report and interpret pseudo R-squared (how does it compare to **m1, m2, m3**?) Comment on the coefficients and denote which ones are significant and insignificant (given their p-values).
12. Submit to GauchoSpace as ipynb/pdf. 