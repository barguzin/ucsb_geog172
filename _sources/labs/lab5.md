# Lab 5: Testing for means, From Correlation and Autocorrelation

````{caution}
This lab will provide an opportunity to practice inferential statistics in Python. Please start by opening [this Google Colab notebook](https://colab.research.google.com/drive/1BMSDi0hpQJ3kVSSNN3a-8n5u05hyXKOC?usp=sharing) and copying it to your Google Drive. 
````

````{important}
Please submit your lab by modifying the code in the in the GC above!
````

````{warning}
Make sure to submit the lab by **October 31 / November 1, 2022**, depending on the scheduled time of your section.
````

---

## Lab 5 Instructions. 

1. Pre-load Guerry data. 
2. Choose one of the **numeric** variables in Guerry geodataframe. See full list of variables [here](https://geodacenter.github.io/data-and-lab/Guerry/). **You CANNOT use any of the variables from demonstration in the lab or in class**. 
3. Check normality assumption using a histogram, QQPlot and Shapiro-Wilk test. 
4. Run a **one sample test** to check whether the sample mean is equal to the mean(varible_of_your_interest). Write down your hypotheses in markdown. Interpret your results. 
5. Group French departments into East and West based on mean longitude (see example code in the lab). 
6. Run a **two sample test** to compare the mean between the East and the West. Write down your hypotheses in markdown. Interpret your results. 
7. Run **ANOVA** to compare the mean of your variable of interest. Write down your hypotheses in markdown. Interpret your results. 
8. Generate **pairwise correlation** between your variable of interest and other numeric variables in the data frame. 
9. Use **Rooks** contingency rule to generate spatial weights matrix for Guerry data. Standardize weights by rows. 
10. Calculate Moran's $I$ for the variable of your choice. 
11. Calculate simulated $p$-value. Interpret your results. 
12. Create Moran's scatter plot for the variable of your choice. Interpet your plot. 
13. Submit your report to GauchoSpace. Please submit both the .ipynb and .pdf of the notebook. 