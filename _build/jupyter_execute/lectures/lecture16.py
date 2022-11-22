#!/usr/bin/env python
# coding: utf-8

# <h1> <center> GEOG 172: INTERMEDIATE GEOGRAPHICAL ANALYSIS </h1>
#     <h2> <center> Evgeny Noi </h2>
#         <h3> <center> Lecture 16: Regression Diagnostics and Spatial Regression </h3>

# # Plan for today 
# 
# 1. Regression Diagnostics 
# 2. Spatial Regression (GWR)

# In[1]:


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt 
import seaborn as sns
import pingouin as pg
import statsmodels
import statsmodels.formula.api as smf
from pysal.model import spreg
from libpysal.weights import Queen, Rook, KNN
from esda.moran import Moran
from pysal.explore import esda


# In[2]:


db = gpd.read_file("regression_db.geojson")
print(db.shape) 
db.head()


# In[3]:


variable_names = [
    "accommodates",  # Number of people it accommodates
    "bathrooms",  # Number of bathrooms
    "bedrooms",  # Number of bedrooms
    "beds",  # Number of beds
    # Below are binary variables, 1 True, 0 False
    "rt_Private_room",  # Room type: private room
    "rt_Shared_room",  # Room type: shared room
    "pg_Condominium",  # Property group: condo
    "pg_House",  # Property group: house
    "pg_Other",  # Property group: other
    "pg_Townhouse",  # Property group: townhouse
]


# In[4]:


fig, ax = plt.subplots(1,2, figsize=(16,4))

ax[0].hist(db.price);
ax[0].set_title('Price', fontsize=14)
ax[1].hist(db.log_price);
ax[1].set_title('log(Price)', fontsize=14)


# In[5]:


# Fitting linear model
res = smf.ols(formula= "log_price ~ accommodates + bathrooms + bedrooms + beds + rt_Private_room + rt_Shared_room +               pg_Condominium + pg_House + pg_Other + pg_Townhouse", data=db).fit()
res.summary()


# In[6]:


# base code
import numpy as np
import seaborn as sns
from statsmodels.tools.tools import maybe_unwrap_results
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from typing import Type
import statsmodels.stats.api as sms

style_talk = 'seaborn-talk'    #refer to plt.style.available

class Linear_Reg_Diagnostic():
    """
    Diagnostic plots to identify potential problems in a linear regression fit.
    Mainly,
        a. non-linearity of data
        b. Correlation of error terms
        c. non-constant variance
        d. outliers
        e. high-leverage points
        f. collinearity

    Author:
        Prajwal Kafle (p33ajkafle@gmail.com, where 3 = r)
        Does not come with any sort of warranty.
        Please test the code one your end before using.
    """

    def __init__(self,
                 results: Type[statsmodels.regression.linear_model.RegressionResultsWrapper]) -> None:
        """
        For a linear regression model, generates following diagnostic plots:

        a. residual
        b. qq
        c. scale location and
        d. leverage

        and a table

        e. vif

        Args:
            results (Type[statsmodels.regression.linear_model.RegressionResultsWrapper]):
                must be instance of statsmodels.regression.linear_model object

        Raises:
            TypeError: if instance does not belong to above object

        Example:
        >>> import numpy as np
        >>> import pandas as pd
        >>> import statsmodels.formula.api as smf
        >>> x = np.linspace(-np.pi, np.pi, 100)
        >>> y = 3*x + 8 + np.random.normal(0,1, 100)
        >>> df = pd.DataFrame({'x':x, 'y':y})
        >>> res = smf.ols(formula= "y ~ x", data=df).fit()
        >>> cls = Linear_Reg_Diagnostic(res)
        >>> cls(plot_context="seaborn-paper")

        In case you do not need all plots you can also independently make an individual plot/table
        in following ways

        >>> cls = Linear_Reg_Diagnostic(res)
        >>> cls.residual_plot()
        >>> cls.qq_plot()
        >>> cls.scale_location_plot()
        >>> cls.leverage_plot()
        >>> cls.vif_table()
        """

        if isinstance(results, statsmodels.regression.linear_model.RegressionResultsWrapper) is False:
            raise TypeError("result must be instance of statsmodels.regression.linear_model.RegressionResultsWrapper object")

        self.results = maybe_unwrap_results(results)

        self.y_true = self.results.model.endog
        self.y_predict = self.results.fittedvalues
        self.xvar = self.results.model.exog
        self.xvar_names = self.results.model.exog_names

        self.residual = np.array(self.results.resid)
        influence = self.results.get_influence()
        self.residual_norm = influence.resid_studentized_internal
        self.leverage = influence.hat_matrix_diag
        self.cooks_distance = influence.cooks_distance[0]
        self.nparams = len(self.results.params)

    def __call__(self, plot_context='seaborn-paper'):
        # print(plt.style.available)
        with plt.style.context(plot_context):
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
            self.residual_plot(ax=ax[0,0])
            self.qq_plot(ax=ax[0,1])
            self.scale_location_plot(ax=ax[1,0])
            self.leverage_plot(ax=ax[1,1])
            plt.show()

        self.vif_table()
        return fig, ax


    def residual_plot(self, ax=None):
        """
        Residual vs Fitted Plot

        Graphical tool to identify non-linearity.
        (Roughly) Horizontal red line is an indicator that the residual has a linear pattern
        """
        if ax is None:
            fig, ax = plt.subplots()

        sns.residplot(
            x=self.y_predict,
            y=self.residual,
            lowess=True,
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
            ax=ax)

        # annotations
        residual_abs = np.abs(self.residual)
        abs_resid = np.flip(np.sort(residual_abs))
        abs_resid_top_3 = abs_resid[:3]
        for i, _ in enumerate(abs_resid_top_3):
            ax.annotate(
                i,
                xy=(self.y_predict[i], self.residual[i]),
                color='C3')

        ax.set_title('Residuals vs Fitted', fontweight="bold")
        ax.set_xlabel('Fitted values')
        ax.set_ylabel('Residuals')
        return ax

    def qq_plot(self, ax=None):
        """
        Standarized Residual vs Theoretical Quantile plot

        Used to visually check if residuals are normally distributed.
        Points spread along the diagonal line will suggest so.
        """
        if ax is None:
            fig, ax = plt.subplots()

        QQ = ProbPlot(self.residual_norm)
        QQ.qqplot(line='45', alpha=0.5, lw=1, ax=ax)

        # annotations
        abs_norm_resid = np.flip(np.argsort(np.abs(self.residual_norm)), 0)
        abs_norm_resid_top_3 = abs_norm_resid[:3]
        for r, i in enumerate(abs_norm_resid_top_3):
            ax.annotate(
                i,
                xy=(np.flip(QQ.theoretical_quantiles, 0)[r], self.residual_norm[i]),
                ha='right', color='C3')

        ax.set_title('Normal Q-Q', fontweight="bold")
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Standardized Residuals')
        return ax

    def scale_location_plot(self, ax=None):
        """
        Sqrt(Standarized Residual) vs Fitted values plot

        Used to check homoscedasticity of the residuals.
        Horizontal line will suggest so.
        """
        if ax is None:
            fig, ax = plt.subplots()

        residual_norm_abs_sqrt = np.sqrt(np.abs(self.residual_norm))

        ax.scatter(self.y_predict, residual_norm_abs_sqrt, alpha=0.5);
        sns.regplot(
            x=self.y_predict,
            y=residual_norm_abs_sqrt,
            scatter=False, ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
            ax=ax)

        # annotations
        abs_sq_norm_resid = np.flip(np.argsort(residual_norm_abs_sqrt), 0)
        abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
        for i in abs_sq_norm_resid_top_3:
            ax.annotate(
                i,
                xy=(self.y_predict[i], residual_norm_abs_sqrt[i]),
                color='C3')
        ax.set_title('Scale-Location', fontweight="bold")
        ax.set_xlabel('Fitted values')
        ax.set_ylabel(r'$\sqrt{|\mathrm{Standardized\ Residuals}|}$');
        return ax

    def leverage_plot(self, ax=None):
        """
        Residual vs Leverage plot

        Points falling outside Cook's distance curves are considered observation that can sway the fit
        aka are influential.
        Good to have none outside the curves.
        """
        if ax is None:
            fig, ax = plt.subplots()

        ax.scatter(
            self.leverage,
            self.residual_norm,
            alpha=0.5);

        sns.regplot(
            x=self.leverage,
            y=self.residual_norm,
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
            ax=ax)

        # annotations
        leverage_top_3 = np.flip(np.argsort(self.cooks_distance), 0)[:3]
        for i in leverage_top_3:
            ax.annotate(
                i,
                xy=(self.leverage[i], self.residual_norm[i]),
                color = 'C3')

        xtemp, ytemp = self.__cooks_dist_line(0.5) # 0.5 line
        ax.plot(xtemp, ytemp, label="Cook's distance", lw=1, ls='--', color='red')
        xtemp, ytemp = self.__cooks_dist_line(1) # 1 line
        ax.plot(xtemp, ytemp, lw=1, ls='--', color='red')

        ax.set_xlim(0, max(self.leverage)+0.01)
        ax.set_title('Residuals vs Leverage', fontweight="bold")
        ax.set_xlabel('Leverage')
        ax.set_ylabel('Standardized Residuals')
        ax.legend(loc='upper right')
        return ax

    def vif_table(self):
        """
        VIF table

        VIF, the variance inflation factor, is a measure of multicollinearity.
        VIF > 5 for a variable indicates that it is highly collinear with the
        other input variables.
        """
        vif_df = pd.DataFrame()
        vif_df["Features"] = self.xvar_names
        vif_df["VIF Factor"] = [variance_inflation_factor(self.xvar, i) for i in range(self.xvar.shape[1])]

        print(vif_df
                .sort_values("VIF Factor")
                .round(2))


    def __cooks_dist_line(self, factor):
        """
        Helper function for plotting Cook's distance curves
        """
        p = self.nparams
        formula = lambda x: np.sqrt((factor * p * (1 - x)) / x)
        x = np.linspace(0.001, max(self.leverage), 50)
        y = formula(x)
        return x, y


# In[7]:


cls = Linear_Reg_Diagnostic(res)

cls.residual_plot()


# In[8]:


cls.qq_plot();


# In[9]:


cls.scale_location_plot();


# In[10]:


# look out for the points outside of the Cook's curves
cls.leverage_plot();


# In[11]:


# look out for the variance inflation factor (VIF) > 5 
cls.vif_table()


# # Getting at Spatial Structure of Data 
# 
# 1. Visual plots 
# 2. Global Moran's I 
# 3. Local Moran's I 

# In[12]:


# Fit OLS model
m1 = spreg.OLS(
    # Dependent variable
    db[["log_price"]].values,
    # Independent variables
    db[variable_names].values,
    # Dependent variable name
    name_y="log_price",
    # Independent variable name
    name_x=variable_names,
)


# In[13]:


# Create column with residual values from m1
db["residual"] = m1.u
# Obtain the median value of residuals in each neighbourhood
medians = (db.groupby("neighborhood").residual.median().to_frame("hood_residual"))

# Increase fontsize
sns.set(font_scale=1.25)
# Set up figure
f = plt.figure(figsize=(15, 3))
# Grab figure's axis
ax = plt.gca()
# Generate bloxplot of values by neighbourhood
data=db.merge(medians, how="left", left_on="neighborhood", right_index=True).sort_values("hood_residual")
# Note the data includes the median values merged on-the-fly
sns.boxplot(
    x="neighborhood",
    y="residual",
    ax=ax,
    data=data,
    palette="bwr"
)
# Auto-format of the X labels
f.autofmt_xdate()
# Display
plt.show()


# In[14]:


from pysal.lib import weights
w_knn = KNN.from_dataframe(db, k=15)

lag_residual = weights.spatial_lag.lag_spatial(w_knn, m1.u)
ax = sns.regplot(
    x=m1.u.flatten(),
    y=lag_residual.flatten(),
    line_kws=dict(color="orangered"),
    ci=None,
)
ax.set_xlabel("Model Residuals - $u$")
ax.set_ylabel("Spatial Lag of Model Residuals - $W u$");


# In[15]:


moran = Moran(res.resid, w_knn)
print(moran.I, moran.p_sim)


# # Interpreting Moran Scatterplot
# 
# > When the model tends to over-predict a given AirBnB’s nightly log price, sites around that AirBnB are more likely to also be over-predicted.

# In[16]:


import contextily
# Re-weight W to 20 nearest neighbors
# knn.reweight(k=20, inplace=True)
# Row standardise weights
w_knn.transform = "R"
# Run LISA on residuals
outliers = esda.moran.Moran_Local(m1.u, w_knn, permutations=9999)
# Select only LISA cluster cores
error_clusters = outliers.q % 2 == 1
# Filter out non-significant clusters
error_clusters &= outliers.p_sim <= 0.001
# Add `error_clusters` and `local_I` columns
ax = (
    db.assign(
        error_clusters=error_clusters,
        local_I=outliers.Is
        # Retain error clusters only
    )
    .query(
        "error_clusters"
        # Sort by I value to largest plot on top
    )
    .sort_values(
        "local_I"
        # Plot I values
    )
    .plot("local_I", cmap="bwr", marker=".")
)
# Add basemap
contextily.add_basemap(ax, crs=db.crs)
# Remove axes
ax.set_axis_off();


# # Interpreting LISA maps 
# 
# > Coldspots - underpredictions, hotspots - overpredictions. We observe spatial patterns to model residuals (predictive performance). 
# 
# #### We need to bring geography into regression framework 

# # Spatial Regression 
# 
# * Explicitly introduces space or geographical context into the statistical framework of a regression
# * Space can act as a reasonable proxy for other factors we cannot but should include in our model.
# * **Construct spatial featrues to bring geography!**
#     * proximity to Balboa Park (tourist attraction) 

# In[17]:


ax = db.plot("d2balboa", marker=".", s=5, legend=True)
contextily.add_basemap(ax, crs=db.crs)
ax.set_axis_off();


# In[18]:


balboa_names = variable_names + ["d2balboa"]
m2 = spreg.OLS(db[["log_price"]].values,db[balboa_names].values,name_y="log_price",name_x=balboa_names,)
pd.DataFrame(
    [[m1.r2, m1.ar2], [m2.r2, m2.ar2]],
    index=["M1", "M2"],
    columns=["R2", "Adj. R2"],
) # the model fit does not change 


# In[19]:


# Set up table of regression coefficients
pd.DataFrame(
    {
        # Pull out regression coefficients and
        # flatten as they are returned as Nx1 array
        "Coeff.": m2.betas.flatten(),
        # Pull out and flatten standard errors
        "Std. Error": m2.std_err.flatten(),
        # Pull out P-values from t-stat object
        "P-Value": [i[1] for i in m2.t_stat],
    },
    index=m2.name_x,
)


# In[20]:


lag_residual = weights.spatial_lag.lag_spatial(w_knn, m2.u)
ax = sns.regplot(
    x=m2.u.flatten(),
    y=lag_residual.flatten(),
    line_kws=dict(color="orangered"),
    ci=None,
)
ax.set_xlabel("Residuals ($u$)")
ax.set_ylabel("Spatial lag of residuals ($Wu$)");


# # Observations About Model 2 
# 
# * We still observe clustering in residuals (something we are not capturing well) - GEO 
# * Positive coefficient for Balboa is not intuitive (we are paying more to stay farther from Balboa)

# # Spatial Heterogeneity 
# 
# * We have assumed that the influence of distance is uniform across the study area, BUT some neighborhoods are systematically more expensive than others. 
# * **parts of the model vary systematically with geography** 

# # Spatial Fixed Effects 
# 
# * Captures the fixed effect through a proxy that influences price (neighborhood) 
# * Create a separate binary variable for each of the $r$ neighborhoods
# 
# $$
# \log{P_i} = \alpha_r + \sum_k \mathbf{X}_{ik}\beta_k  + \epsilon_i
# $$
# 
# where $\alpha_r$ varies across neighborhoods

# In[21]:


f = (
    "log_price ~ "
    + " + ".join(variable_names)
    + " + neighborhood - 1"
)
print(f)


# In[22]:


m3 = smf.ols(f, data=db).fit()


# In[23]:


# Store variable names for all the spatial fixed effects
sfe_names = [i for i in m3.params.index if "neighborhood[" in i]
# Create table
pd.DataFrame(
    {
        "Coef.": m3.params[sfe_names],
        "Std. Error": m3.bse[sfe_names],
        "P-Value": m3.pvalues[sfe_names],
    }
).head(10)


# In[24]:


# spreg spatial fixed effect implementation
m4 = spreg.OLS_Regimes(
    # Dependent variable
    db[["log_price"]].values,
    # Independent variables
    db[variable_names].values,
    # Variable specifying neighborhood membership
    db["neighborhood"].tolist(),
    # Allow the constant term to vary by group/regime
    constant_regi="many",
    # Variables to be allowed to vary (True) or kept
    # constant (False). Here we set all to False
    cols2regi=[False] * len(variable_names),
    # Allow separate sigma coefficients to be estimated
    # by regime (False so a single sigma)
    regime_err_sep=False,
    # Dependent variable name
    name_y="log_price",
    # Independent variables names
    name_x=variable_names,
)


# In[25]:


np.round(m4.betas.flatten() - m3.params.values, decimals=12)


# In[26]:


neighborhood_effects = m3.params.filter(like="neighborhood")

# Create a sequence with the variable names without
# `neighborhood[` and `]`
stripped = neighborhood_effects.index.str.strip(
    "neighborhood["
).str.strip("]")
# Reindex the neighborhood_effects Series on clean names
neighborhood_effects.index = stripped
# Convert Series to DataFrame
neighborhood_effects = neighborhood_effects.to_frame("fixed_effect")
# Print top of table
neighborhood_effects.head()


# In[27]:


import urllib.request
urllib.request.urlretrieve("http://data.insideairbnb.com/united-states/ca/san-diego/2022-09-18/visualisations/neighbourhoods.geojson ", "sd_geo.geojson")


# In[28]:


neighborhoods = gpd.read_file('sd_geo.geojson')

# Plot base layer with all neighborhoods in grey
ax = neighborhoods.plot(
    color="k", linewidth=0, alpha=0.5, figsize=(12, 6), legend=True
)
# Merge SFE estimates (note not every polygon
# receives an estimate since not every polygon
# contains AirBnb properties)
neighborhoods.merge(
    neighborhood_effects,
    how="left",
    left_on="neighbourhood",
    right_index=True
    # Drop polygons without a SFE estimate
).dropna(
    subset=["fixed_effect"]
    # Plot quantile choropleth
).plot(
    "fixed_effect",  # Variable to display
    scheme="quantiles",  # Choropleth scheme
    k=7,  # No. of classes in the choropleth
    linewidth=0.1,  # Polygon border width
    cmap="viridis",  # Color scheme
    ax=ax,  # Axis to draw on
    legend=True, 
    legend_kwds={'loc': 'center left', 'bbox_to_anchor':(1,0.5)}
)
# Add basemap
contextily.add_basemap(
    ax,
    crs=neighborhoods.crs,
    source=contextily.providers.CartoDB.PositronNoLabels,
)
# Remove axis
ax.set_axis_off()
# Display
plt.show()


# # Interpeting Spatial Fixed Effects 
# 
# > We can see a clear spatial structure in the SFE estimates. The most expensive neighborhoods tend to be located nearby the coast, while the cheapest ones are more inland.

# In[29]:


lag_residual = weights.spatial_lag.lag_spatial(w_knn, m4.u)
ax = sns.regplot(
    x=m4.u.flatten(),
    y=lag_residual.flatten(),
    line_kws=dict(color="orangered"),
    ci=None,
)
ax.set_xlabel("Residuals ($u$)")
ax.set_ylabel("Spatial lag of residuals ($Wu$)");


# In[30]:


moran = Moran(m4.u, w_knn)
print(moran.I, moran.p_sim) # smaller Moran's I (compared to .14) 


# # Spatial Regimes 
# 
# > Allow not only intercepts, but beta coefficients to vary 
# 
# $$
# \log{P_i} = \alpha_r + \sum_k \mathbf{X}_{ki}\beta_{k-r} + \epsilon_i
# $$

# In[31]:


# Pysal spatial regimes implementation
m5 = spreg.OLS_Regimes(
    # Dependent variable
    db[["log_price"]].values,
    # Independent variables
    db[variable_names].values,
    # Variable specifying neighborhood membership
    db["coastal"].tolist(),
    # Allow the constant term to vary by group/regime
    constant_regi="many",
    # Allow separate sigma coefficients to be estimated
    # by regime (False so a single sigma)
    regime_err_sep=False,
    # Dependent variable name
    name_y="log_price",
    # Independent variables names
    name_x=variable_names,
)


# In[32]:


# Results table
res = pd.DataFrame({
        "Coeff.": m5.betas.flatten(),
        "Std. Error": m5.std_err.flatten(),
        "P-Value": [i[1] for i in m5.t_stat],},
    index=m5.name_x,
)

coastal = [i for i in res.index if "1_" in i]
coastal = res.loc[coastal, :].rename(lambda i: i.replace("1_", ""))

coastal.columns = pd.MultiIndex.from_product([["Coastal"], coastal.columns])

ncoastal = [i for i in res.index if "0_" in i]

ncoastal = res.loc[ncoastal, :].rename(lambda i: i.replace("0_", ""))

ncoastal.columns = pd.MultiIndex.from_product([["Non-coastal"], ncoastal.columns])

# Concat both models
pd.concat([coastal, ncoastal], axis=1)


# In[33]:


m5.chow.joint


# In[34]:


pd.DataFrame(
    # Chow results by variable
    m5.chow.regi,
    # Name of variables
    index=m5.name_x_r,
    # Column names
    columns=["Statistic", "P-value"],
)


# In[35]:


lag_residual = weights.spatial_lag.lag_spatial(w_knn, m5.u)
ax = sns.regplot(
    x=m5.u.flatten(),
    y=lag_residual.flatten(),
    line_kws=dict(color="orangered"),
    ci=None,
)
ax.set_xlabel("Residuals ($u$)")
ax.set_ylabel("Spatial lag of residuals ($Wu$)");


# In[36]:


moran = Moran(m5.u, w_knn)
print(moran.I, moran.p_sim) # smaller Moran's I (compared to m2 - .14, but larger than m4 - .4) 


# # Spatial Dependence 
# 
# * We shift our interest from explicitly linked geography, to **spatial configuration**. Think about neighborhood as generalization of surroundings. The look and feel of the neighborhood might depend on how some of them vary in terms of nearby configurations (townhouse next to other townhouses). 
# * **Spatial dependence** through distance
#     * SLX model 
#     * Spatial Error
#     * Spatial Lag 

# # SLX Model (Exogenous effects) 
# 
# $$
# \log(P_i) = \alpha + \sum^{p}_{k=1}X_{ij}\beta_j + \sum^{p}_{k=1}\left(\sum^{N}_{j=1}w_{ij}x_{jk}\right)\gamma_k + \epsilon_i
# $$
# 
# $$
# \log(P_i) = \alpha + \mathbf{X}\beta + \mathbf{WX}\gamma + \epsilon
# $$
# 
# * We separate spatial effects into direct ($\beta$) and indirect ($\gamma$). 

# # SLX Model (Example) 
# 
# * Add spatial lag of type of houses in the neighborhood (surrounding) 
# * Association between the price in a given house and a unit change in its average surroundings 
# * Spillover effect

# In[37]:


# Select only columns in `db` containing the keyword `pg_`
wx = (
    db.filter(
        like="pg_"
        # Compute the spatial lag of each of those variables
    )
    .apply(
        lambda y: weights.spatial_lag.lag_spatial(w_knn, y)
        # Rename the spatial lag, adding w_ to the original name
    )
    .rename(
        columns=lambda c: "w_"
        + c
        # Remove the lag of the binary variable for apartments
    )
    .drop("w_pg_Apartment", axis=1)
)


# In[38]:


# Merge original variables with the spatial lags in `wx`
slx_exog = db[variable_names].join(wx)
# Fit linear model with `spreg`
m6 = spreg.OLS(
    # Dependent variable
    db[["log_price"]].values,
    # Independent variables
    slx_exog.values,
    # Dependent variable name
    name_y="l_price",
    # Independent variables names
    name_x=slx_exog.columns.tolist(),
)


# In[39]:


# Collect names of variables of interest
vars_of_interest = (
    db[variable_names].filter(like="pg_").join(wx).columns
)
# Build full table of regression coefficients
pd.DataFrame(
    {
        # Pull out regression coefficients and
        # flatten as they are returned as Nx1 array
        "Coeff.": m6.betas.flatten(),
        # Pull out and flatten standard errors
        "Std. Error": m6.std_err.flatten(),
        # Pull out P-values from t-stat object
        "P-Value": [i[1] for i in m6.t_stat],
    },
    index=m6.name_x
    # Subset for variables of itnerest only and round to
    # four decimals
).reindex(vars_of_interest).round(4)


# # Interpreting SLX model 
# 
# * The direct effect of the pg_Condominium variable means that condominiums are typically 11% more expensive than benchmark property type (apartments) 
# * Since *pg_Condominium* is a dummy variable, the spatial lag at site $i$ represents the percentage of properties near $i$ that are condominiums. So a unit change in this variable means that you would increase the condominium percentage by 100%. Thus a .1 increase in *w_pg_Condominium* (a change of ten percentage points) would result in 4.93% increase in the property house price.  

# In[40]:


moran = Moran(m6.u, w_knn)
print(moran.I, moran.p_sim) # smaller Moran's I (compared to m2 - .14, but larger than m4 - .4) 
lag_residual = weights.spatial_lag.lag_spatial(w_knn, m6.u)
ax = sns.regplot(
    x=m6.u.flatten(),
    y=lag_residual.flatten(),
    line_kws=dict(color="orangered"),
    ci=None,
)
ax.set_xlabel("Residuals ($u$)")
ax.set_ylabel("Spatial lag of residuals ($Wu$)");


# # Spatial Error 
# 
# > Include spatial lag in the error term of the equation. This violates OLS assumption (normal errors).  
# 
# $$
# \log{P_i} = \alpha + \sum_k \beta_k X_{ki} + u_i \\
# u_i = \lambda u_{lag-i} + \epsilon_i
# $$
# 
# where $u_{lag-i} = \sum_j w_{i,j} u_j$. 

# In[41]:


# Fit spatial error model with `spreg`
# (GMM estimation allowing for heteroskedasticity)
m7 = spreg.GM_Error_Het(
    # Dependent variable
    db[["log_price"]].values,
    # Independent variables
    db[variable_names].values,
    # Spatial weights matrix
    w=w_knn,
    # Dependent variable name
    name_y="log_price",
    # Independent variables names
    name_x=variable_names,
)


# In[42]:


print(m7.summary)


# In[43]:


moran = Moran(m7.u, w_knn)
print(moran.I, moran.p_sim) # smaller Moran's I (compared to m2 - .14, but larger than m4 - .4) 
lag_residual = weights.spatial_lag.lag_spatial(w_knn, m7.u)
ax = sns.regplot(
    x=m7.u.flatten(),
    y=lag_residual.flatten(),
    line_kws=dict(color="orangered"),
    ci=None,
)
ax.set_xlabel("Residuals ($u$)")
ax.set_ylabel("Spatial lag of residuals ($Wu$)");


# # Spatial Lag 
# 
# > Spatial lag of the **dependent varialbe**
# 
# $$
# \log{P_i} = \alpha + \rho \log{P_{lag-i}} + \sum_k \beta_k X_{ki} + \epsilon_i
# $$

# In[44]:


# Fit spatial lag model with `spreg`
# (GMM estimation)
m8 = spreg.GM_Lag(
    # Dependent variable
    db[["log_price"]].values,
    # Independent variables
    db[variable_names].values,
    # Spatial weights matrix
    w=w_knn,
    # Dependent variable name
    name_y="log_price",
    # Independent variables names
    name_x=variable_names,
)


# In[45]:


# Build full table of regression coefficients
pd.DataFrame(
    {
        # Pull out regression coefficients and
        # flatten as they are returned as Nx1 array
        "Coeff.": m8.betas.flatten(),
        # Pull out and flatten standard errors
        "Std. Error": m8.std_err.flatten(),
        # Pull out P-values from t-stat object
        "P-Value": [i[1] for i in m8.z_stat],
    },
    index=m8.name_z
    # Round to four decimals
).round(4)


# In[46]:


moran = Moran(m8.u, w_knn)
print(moran.I, moran.p_sim) # smaller Moran's I (compared to m2 - .14, but larger than m4 - .4) 
lag_residual = weights.spatial_lag.lag_spatial(w_knn, m8.u)
ax = sns.regplot(
    x=m8.u.flatten(),
    y=lag_residual.flatten(),
    line_kws=dict(color="orangered"),
    ci=None,
)
ax.set_xlabel("Residuals ($u$)")
ax.set_ylabel("Spatial lag of residuals ($Wu$)");


# In[47]:


# compare models 
pd.DataFrame(
    {
        'models': ['spatial fixed effects', 'spatial regimes', 'SLX model', 'Spatial Error', 'Spatial Lag'],
        'rsq': [m4.r2, m5.r2, m6.r2, 0.66, 0.7052], 
    })


# # Sources 
# 
# [Geographic Data Science Book - Chapter 11](https://geographicdata.science/book/notebooks/11_regression.html#)
# 
# Anselin, Luc and Sergio Rey. 2014. Modern Spatial Econometrics in Practice: A Guide to GeoDa, GeoDaSpace, and Pysal. GeoDa Press.
# 
# 

# # Questions? 
