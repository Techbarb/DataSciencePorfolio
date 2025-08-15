#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("World Energy Consumption.csv")


# In[3]:


print(df.head(10))


# In[4]:


#Clean the data


# In[5]:


print(df.columns)


# In[6]:


columns_to_keep = [
    # Biofuel / renewables
   "country", "year", "population", "gdp", "biofuel_cons_change_pct", "biofuel_cons_change_twh", "biofuel_consumption", "biofuel_electricity", "biofuel_share_elec", "biofuel_share_energy",
    "low_carbon_consumption", "low_carbon_electricity", "low_carbon_share_elec",
    "renewables_consumption", "renewables_electricity", "renewables_share_elec",
    "solar_consumption", "solar_electricity",
    "wind_consumption", "wind_electricity",
    "other_renewable_consumption",

    # Fossil fuels
    "coal_consumption", "coal_electricity",
    "gas_consumption", "gas_electricity",
    "oil_consumption", "oil_electricity",
    "fossil_fuel_consumption", "fossil_electricity",

    # Total energy / efficiency
    "primary_energy_consumption", "energy_cons_change_pct", "energy_cons_change_twh",
    "energy_per_capita", "energy_per_gdp",
    "electricity_generation", "electricity_demand",

    # Environmental impact
    "carbon_intensity_elec", "greenhouse_gas_emissions",

    # Optional: imports
    "net_elec_imports", "net_elec_imports_share_demand"
]


# In[7]:


df_clean = df[columns_to_keep]


# In[8]:


print(df_clean.head(10))
print(df_clean.columns)


# In[9]:


df_clean1 = df_clean.dropna()
print(df_clean1.head(10))


# EXPLORATORY DATA ANALYSIS

# In[10]:


print(df_clean1.columns)


# In[11]:


# Get the latest year in your data
latest_year = df_clean1['year'].max()
print(latest_year)


# In[12]:


# Filter to only keep rows from the past 3 years
df_last_3_years = df_clean1[df_clean1['year'] >= latest_year - 2]

print(df_last_3_years['year'].unique())
print(df_last_3_years.head())


# In[16]:


print(df_last_3_years.shape)

print(df_last_3_years.dtypes)



# In[19]:


latest_year = df_last_3_years['year'].max()  # gets the max year
df_latest = df_last_3_years[df_last_3_years['year'] == latest_year]


# In[20]:


#Top 5 countries with the highest energy consumption
top5 = df_latest.nlargest(5, 'primary_energy_consumption')
print(top5)


# In[21]:


plt.figure(figsize=(10,6))  # set figure size
plt.bar(top5['country'], top5['primary_energy_consumption'], color='purple')

plt.xlabel('Country')
plt.ylabel('Primary Energy Consumption (TWh)')
plt.title(f'Top 5 Countries by Primary Energy Consumption in {latest_year}')
plt.xticks(rotation=45)  # rotate labels for better readability
plt.show()

 


# In[23]:


#5 countries with the lowest energy consumption

least5 = df_latest.nsmallest(5, 'primary_energy_consumption')
print(least5)


# In[25]:


plt.figure(figsize=(10,6))
plt.bar(least5['country'], least5['primary_energy_consumption'], color='blue')
plt.xlabel('Country')
plt.ylabel('Primary Energy Consumption')
plt.title(f'Least 5 Countries by Primary Energy Consumption in {latest_year}')
plt.xticks(rotation=45)  # rotate labels for better readability
plt.show()


# In[26]:


#gdp analysis 
gdptop5 =  df_latest.nlargest(5, 'gdp')
print(gdptop5 )


# In[28]:


plt.figure(figsize=(10,6))
plt.bar(gdptop5['country'], gdptop5['gdp'], color='pink')
plt.xlabel('Country')
plt.ylabel('GDP')
plt.title(f'Top 5 Countries by GDP in {latest_year}')
plt.xticks(rotation=45)  # rotate labels for better readability
plt.show()




# In[29]:


gdpleast5 =  df_latest.nsmallest(5, 'gdp')
print(gdpleast5) 


# In[31]:


plt.figure(figsize=(10,6))
plt.bar(gdpleast5['country'], gdpleast5['gdp'], color='green')
plt.xlabel('Country')
plt.ylabel('GDP')
plt.title(f'Least 5 Countries by GDP in {latest_year}')
plt.xticks(rotation=45)  # rotate labels for better readability
plt.show()


# In[32]:


# 1. Filter dataset for the last 3 years and top 5 countries
top5_countries = top5['country'].tolist()
df_top5_last3 = df_last_3_years[df_last_3_years['country'].isin(top5_countries)]

# 2. Create the line chart
plt.figure(figsize=(10,6))

for country in top5_countries:
    country_data = df_top5_last3[df_top5_last3['country'] == country]
    plt.plot(country_data['year'], country_data['gdp'], marker='o', label=country)

# 3. Add labels and title
plt.xlabel('Year')
plt.ylabel('GDP (in billions or trillions)')
plt.title('GDP of Top 5 Countries over the Last 3 Years')

# 4. Add legend
plt.legend()

# 5. Show chart
plt.show()






# In[34]:


# 1. Pick the countries you want to compare
selected_countries = ['Nigeria', 'United Kingdom', 'United States', 'China']  # replace with your choices

# 2. Filter the dataframe for those countries
df_selected = df_last_3_years[df_last_3_years['country'].isin(selected_countries)]

# 3. Create the line chart
plt.figure(figsize=(10,6))

for country in selected_countries:
    country_data = df_selected[df_selected['country'] == country]
    plt.plot(country_data['year'], 
             country_data['primary_energy_consumption'], 
             marker='o', 
             label=country)

# 4. Labels & Title
plt.xlabel('Year')
plt.ylabel('Primary Energy Consumption (TWh)')
plt.title('Primary Energy Consumption Increase for Selected Countries')

# 5. Legend
plt.legend()

# 6. Show chart
plt.show()


# In[52]:


#Energy consumption vs energy demand for 

top_demand_countries = (
    df_last_3_years.groupby('country')['electricity_demand']
    .sum()
    .nlargest(5)  # pick top 5
    .index
)

print(top_demand_countries)


# In[53]:


df_top_demand = df_last_3_years[df_last_3_years['country'].isin(top_demand_countries)]

print(df_top_demand)


# In[55]:


plt.figure(figsize=(8,6))
plt.scatter(
    df_top_demand['electricity_demand'],
    df_top_demand['electricity_generation'],
    alpha=0.7,
    color='red'
)

# Add labels to each point
for i, row in df_top_demand.iterrows():
    plt.text(
        row['electricity_demand'], 
        row['electricity_generation'], 
        row['country'], 
        fontsize=9
    )

plt.xlabel('Electricity Demand (TWh)')
plt.ylabel('Electricity Generation')
plt.title('Electricity Demand vs Electricity Generation (Top Demand Countries)')
plt.show()


# In[57]:


#GDP vs Population

# 1️⃣ Get top 5 countries by population
top_population_countries = (
    df_last_3_years.groupby('country')['population']
    .max()  # in case you have multiple years per country
    .nlargest(5)
    .index
)

# 2️⃣ Filter dataset
df_top_pop = df_last_3_years[df_last_3_years['country'].isin(top_population_countries)]

# 3️⃣ Plot
plt.figure(figsize=(8,6))
plt.scatter(df_top_pop['population'], df_top_pop['gdp'], alpha=0.7, color='green')

# Add labels
for i, row in df_top_pop.iterrows():
    plt.text(row['population'], row['gdp'], row['country'], fontsize=9)

plt.xlabel('Population')
plt.ylabel('GDP (in USD)')
plt.title('GDP vs Population (Top Population Countries)')
plt.xscale('log')
plt.yscale('log')
plt.show()


# # MODEL BUILDING 

# In[60]:


#seperate into categories
target = ['primary_energy_consumption']


# In[80]:


features = [
    'population',
    'gdp',
    'biofuel_consumption',
    'low_carbon_consumption',
    'renewables_consumption',
    'solar_consumption',
    'wind_consumption',
    'other_renewable_consumption',
    'coal_consumption',
    'gas_consumption',
    'oil_consumption',
    'fossil_fuel_consumption'
   
]


# In[81]:


#Splitting the model into train and testing
from sklearn.model_selection import train_test_split

X = df_last_3_years[features]
y = df_last_3_years[target]

# Drop rows with NaNs in either features or target
X = X.dropna()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[82]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)


# In[83]:


from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Predictions
y_pred = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R² Score:", r2)
print("RMSE:", rmse)


# In[85]:


from sklearn.metrics import precision_score, recall_score

# Example: classify as 1 if predicted value >= 0.5, else 0
y_pred_class = (y_pred >= 0.5).astype(int)
y_true_class = (y_test >= 0.5).astype(int)

precision = precision_score(y_true_class, y_pred_class)
recall = recall_score(y_true_class, y_pred_class)

print("Precision:", precision)
print("Recall:", recall)


# In[71]:


get_ipython().system('pip install shap')


# In[86]:


import shap

# Create SHAP explainer
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Summary plot (requires matplotlib)
shap.summary_plot(shap_values, X_test)

# Beeswarm plot (feature importance & effect)
shap.plots.beeswarm(shap_values)

# Bar plot for mean absolute SHAP values
shap.plots.bar(shap_values)


# # RANDOM FOREST

# In[88]:


# If y is a DataFrame
y = y.values.ravel()  # converts to 1D array

# or
y = df_last_3_years['primary_energy_consumption']  # select as Series


# In[89]:


from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)


# In[90]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np


# In[91]:


r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"R² Score: {r2}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")


# In[93]:


import shap

# Create SHAP explainer
explainer = shap.Explainer(rf_model, X_train)
shap_values = explainer(X_test)

# Beeswarm plot (feature importance & effect)
shap.plots.beeswarm(shap_values)

# Bar plot for mean absolute SHAP values
shap.plots.bar(shap_values)


# # XGBoost
# 

# In[94]:


get_ipython().system('pip install xgboost')


# In[95]:


from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)



# In[96]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"R² Score: {r2}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")


# In[98]:


import shap

# Create SHAP explainer
explainer = shap.Explainer(xgb_model, X_train)
shap_values = explainer(X_test)

# Beeswarm plot (feature importance & effect)
shap.plots.beeswarm(shap_values)

# Bar plot for mean absolute SHAP values
shap.plots.bar(shap_values)



# # RESIDUAL ANALYSIS

# In[100]:


# If y_test is a DataFrame, convert to Series
y_test_series = y_test.squeeze()  # converts single-column DataFrame to Series


# In[101]:


residuals = y_test_series - y_pred  # now dimensions match


# In[102]:


import matplotlib.pyplot as plt

# Predicted vs Actual
plt.figure(figsize=(8,6))
plt.scatter(y_test_series, y_pred, alpha=0.7)
plt.plot([y_test_series.min(), y_test_series.max()],
         [y_test_series.min(), y_test_series.max()], 'r--')
plt.xlabel('Actual Energy Consumption')
plt.ylabel('Predicted Energy Consumption')
plt.title('Predicted vs Actual')
plt.show()

# Residuals
plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')
plt.show()



# In[ ]:


# Assuming you have trained these models: lr_model, rf_model, xgb_model

y_pred = lr_model.predict(X_test)
y_pred = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)


# In[103]:


plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_lr, alpha=0.5, label='Linear Regression')
plt.scatter(y_test, y_pred_rf, alpha=0.5, label='Random Forest')
plt.scatter(y_test, y_pred_xgb, alpha=0.5, label='XGBoost')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel('Actual Energy Consumption')
plt.ylabel('Predicted Energy Consumption')
plt.title('Model Comparison: Predicted vs Actual')
plt.legend()
plt.show()



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




