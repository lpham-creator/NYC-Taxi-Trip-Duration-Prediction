# %% [markdown]
# # NYC TAXI TRIP DURATION PROJECT
# 
# by Linh Pham
# 
# ## 1. Information 
# 
# - Data Source: The dataset is based on the **2016 NYC Yellow Cab trip record** data made available in **Big Query** on **Google Cloud Platform**. The data was originally published by the **NYC Taxi and Limousine Commission (TLC)**. The data was sampled and cleaned for the purposes of this playground. Based on individual trip attributes, should predict the duration of each trip.
# 
# **NYC Taxi and Limousine Commission (TLC)** : http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml
# 
# **Download Data :** https://drive.google.com/open?id=1OyOC9y2x4uyT7drXJBOEZ2yRBktiQB8H
# 
# **Kaggle** : https://www.kaggle.com/c/nyc-taxi-trip-duration/data
# 
# - Data Attributes: 
# 
# ●	train.csv - the dataset (contains 1458644 trip records)
# 
# ●	id - a unique identifier for each trip
# 
# ●	vendor_id - a code indicating the provider associated with the trip record
# 
# ●	pickup_datetime - date and time when the meter was engaged
# 
# ●	dropoff_datetime - date and time when the meter was disengaged
# 
# ●	passenger_count - the number of passengers in the vehicle (driver entered value)
# 
# ●	pickup_longitude - the longitude where the meter was engaged
# 
# ●	pickup_latitude - the latitude where the meter was engaged
# 
# ●	dropoff_longitude - the longitude where the meter was disengaged
# 
# ●	dropoff_latitude - the latitude where the meter was disengaged
# 
# ●	store_and_fwd_flag - This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip
# 
# ●	trip_duration - duration of the trip in seconds
# 
# - Evaluation Metrics: 
# 
# The evaluation metric for this competition is Root Mean Squared Logarithmic Error. The RMSLE is calculated as
# 
# ![alt text](https://i.stack.imgur.com/952Ox.png)
# 
# Where:
# 
# ●	ϵ is the RMSLE value (score)
# 
# ●	n is the total number of observations in the (public/private) data set
# 
# ●	pi is your prediction of trip duration
# 
# ●	ai is the actual trip duration for i. 
# 
# ●	log(x) is the natural logarithm of x
# 
# - Objective: 
# 
# My objective for this project is to explore various attributes and build a Predictive model that predicts the total trip duration of taxi trips in New York City.

# %% [markdown]
# ## 2. Loading Dataset

# %%
#Loading libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('train.csv', header = 0, parse_dates= True)
df.head()

# %%
# Shape of data:

print('No. of examples', df.shape[0])
print('No. of features', df.shape[1])

# %%
# Checking for null values
df.info()

# %%
df.isnull().sum()

# %% [markdown]
# We don't have any null values, so we don't have to work on missing values for this dataset.

# %% [markdown]
# # 3. Early Data Exploration & Processing

# %% [markdown]
# ## 3.1. Categorical Variables

# %%
#Vendor ID
# Count occurrences
vendor_counts = df.vendor_id.value_counts().sort_index()

# Plot bar chart
plt.figure(figsize=(6, 4))
sns.barplot(x=vendor_counts.index, y=vendor_counts.values, palette='viridis')

# Labels and title
plt.xlabel('Vendor ID')
plt.ylabel('Count')
plt.title('Distribution of Vendor IDs')
plt.xticks([0, 1], ['Vendor 1', 'Vendor 2'])  # Rename ticks if needed

# Show the plot
plt.show()

# %% [markdown]
# 🔎 Insights: There are only two types of vendor, and New Yorkers seem to prefer vendor 2 over vendor 1.

# %%
#Passenger count
# Count occurrences
passenger_counts = df.passenger_count.value_counts().sort_index()

# Plot bar chart
plt.figure(figsize=(6, 4))
sns.barplot(x=passenger_counts.index, y=passenger_counts.values, palette='viridis')

# Labels and title
plt.xlabel('Passenger Count')
plt.ylabel('Count')
plt.title('Distribution of Passenger Count')

# Show the plot
plt.show()

# %%
print(passenger_counts)

# %% [markdown]
# 🔎 Insights: The most common trips are those with a single passenger, followed by trips with 2, 3, and 4 passengers. 
# While the maximum recorded passenger count is 9, it's unclear whether vehicles with such a large capacity actually exist. 
# Additionally, since only five trips had 7 or more passengers, these could likely be considered outliers.
# 
# There are 60 trips with 0 passengers, which might be an error in the logging system, so we would also rule these as outliers.

# %%
#Store & Forward flag

plt.figure(figsize=(8,8))
plt.pie(df['store_and_fwd_flag'].value_counts(), colors=['lightgreen', 'lightcoral'], shadow=True, explode=[0.5,0], autopct='%1.2f%%', startangle=200)
plt.legend(labels=['Y','N'])
plt.title("Store and Forward Flag")

# %% [markdown]
# 🔎 Insights: This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y = store and forward; N = not a store and forward trip. The plot shows us that almost all trips were stored in the vehicle before being sent to the vendor, indicating frequent connectivity issues or a system designed to buffer trip data before transmission.
# 
# Potential Network Gaps: The high percentage of 'Y' suggests that the vehicles often operate in areas with poor or no network coverage, leading to delayed data reporting.

# %%
#Label Encoding Features having Categorical Values

from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()
df['store_and_fwd_flag'] = enc.fit_transform(df['store_and_fwd_flag'])
df['vendor_id'] = enc.fit_transform(df['vendor_id'])

#df['vendor_id'] = df['vendor_id'].astype('category')
#df['store_and_fwd_flag'] = df['store_and_fwd_flag'].astype('category')

# %% [markdown]
# 🔎 Insights: Conversion of 'store_and_fwd_flag' and 'vendor_id' to be Label encoded as those are Categorical features , binarizing them will help us to compute them with ease. We can convert these features into “category” type by function called “astype(‘category’)” that will speed up the Computation. Since, my plan is to go with PCA for dimension reduction, I’m not going with that approach.

# %% [markdown]
# ## 3.2. Numerical Variables

# %%
df.describe()[1:]

# %% [markdown]
# 🔎 Insights: At first glance, the trips are short and the pickup - dropoff longitude and latitude are close, indicating there is a large number of short trips. 

# %%
#Visualising Trip Duration
plt.figure(figsize=(20,5))
sns.boxplot(df['trip_duration'])

# %%
df['trip_duration'].max()

# %%
(df['trip_duration'] > 36000).value_counts()

# %%
df['trip_duration'].min()

# %% [markdown]
# 🔎 Insights: There are some outliers here, some trips have the duration of 3526282 seconds - which is just impossible. Some trips only last for 1 second. I'm going to rule out those trips as errors in the logging system - keeping only trips lasting under 10 hours and trips lasting longer than 4 minutes.

# %%
#Log Transformation
plt.figure(figsize=(10,8))
sns.distplot(np.log(df['trip_duration']), kde=False, color='black')
plt.title("Log Transformed - Trip Duration")

# %% [markdown]
# Since our Evaluation Metric is RMSLE, we'll proceed further with Log Transformed "Trip duration". Log Transformation Smoothens outliers by proving them less weightage.

# %% [markdown]
# ## 3.3 Feature Engineering 

# %%
df.pickup_datetime = pd.to_datetime(df.pickup_datetime)
df.dropoff_datetime = pd.to_datetime(df.dropoff_datetime)

df['pickup_day'] = df['pickup_datetime'].dt.day
df['pickup_month'] = df['pickup_datetime'].dt.month
df['pickup_date'] = df['pickup_datetime'].dt.date
df['pickup_hour'] = df['pickup_datetime'].dt.hour
df['pickup_min'] = df['pickup_datetime'].dt.minute
df['pickup_weekday'] = df['pickup_datetime'].dt.weekday

df['dropoff_min'] = df['dropoff_datetime'].dt.minute

# %%
df.pickup_month.value_counts()

# %%
#Pickup Month 
pickup_counts = df.pickup_month.value_counts().sort_index()

# Plot bar chart
plt.figure(figsize=(6, 4))
sns.barplot(x=pickup_counts.index, y=pickup_counts.values, palette='viridis')

# Labels and title
plt.xlabel('Pickup Month')
plt.ylabel('Count')
plt.title('Distribution of Pickup Month')

# Show the plot
plt.show()

# %% [markdown]
# 🔎 Insights: The distribution of pickup month seems pretty balanced. During March and April there are more trips compared to other months. 

# %%
#Pickup Month 
pickup_counts = df.pickup_hour.value_counts().sort_index()

# Plot bar chart
plt.figure(figsize=(8, 4))
sns.barplot(x=pickup_counts.index, y=pickup_counts.values, palette='viridis')

# Labels and title
plt.xlabel('Pickup Month')
plt.xticks(rotation=-45)
plt.ylabel('Count')
plt.title('Distribution of Pickup Hour')

# Show the plot
plt.show()

# %% [markdown]
# 🔎 Insights: People tend to order cabs during evening hour, and the least busy hours are the midnight-early morning hours.

# %%
#Pickup Weekday
pickup_counts = df.pickup_weekday.value_counts().sort_index()

# Plot bar chart
plt.figure(figsize=(20, 4))
sns.barplot(x=pickup_counts.index, y=pickup_counts.values, palette='viridis')

# Labels and title
plt.xlabel('Pickup Weekday')
plt.ylabel('Count')
plt.xticks([0, 1, 2, 3, 4, 5, 6], ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title('Distribution of Pickup Weekday')

# Show the plot
plt.show()

# %%
#Pickup Day
pickup_counts = df.pickup_day.value_counts().sort_index()

# Plot bar chart
plt.figure(figsize=(20, 4))
sns.barplot(x=pickup_counts.index, y=pickup_counts.values, palette='viridis')

# Labels and title
plt.xlabel('Pickup Weekday')
plt.ylabel('Count')
#plt.xticks([0, 1, 2, 3, 4, 5, 6], ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title('Distribution of Pickup Day')

# Show the plot
plt.show()

# %% [markdown]
# 🔎 Insights: Thursday, Friday, and Saturday are the busier days of the week for cabs.

# %%
# Select only numeric columns
df_numeric = df.select_dtypes(include=['number'])

# Compute correlation and scale to percentage
correlation_matrix = df_numeric.corr() * 100

# Plot correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='inferno', fmt=".1f", linewidths=0.5)

# Title
plt.title('Pearson Correlation Heatmap', fontsize=14)

# Show plot
plt.show()


# %% [markdown]
# 🔎 Insights: No pair of variables shows excessive correlation except for pickup - dropoff datetime, so I'll drop the dropoff datetime. The features I just created from pickup - dropoff time are all informative, so I won't drop any of them.

# %% [markdown]
# ## 3.4 Data Cleaning
# 
# We clean the data based on the outliers we observed earlier.

# %%
import pandas as pd

def filter_trip_data(df):
    """
    Remove trips based on the following conditions:
    - Trip duration greater than 10 hours (36000 seconds)
    - Trip duration less than 5 minutes (300 seconds)
    - Trips with more than 7 passengers
    - Trips with 0 passengers
    """
    # Remove trips longer than 10 hours (36000 seconds) and shorter than 5 minutes (300 seconds)
    df_filtered = df[(df['trip_duration'] <= 36000) & (df['trip_duration'] >= 300)]
    
    # Remove trips with more than 7 passengers or 0 passengers
    df_filtered = df_filtered[(df_filtered['passenger_count'] <= 7) & (df_filtered['passenger_count'] > 0)]
    
    return df_filtered

# Apply the function to filter data
filtered_df = filter_trip_data(df)

# %%
# Check our data
filtered_df.head()

# %%
filtered_df.shape[0]/df.shape[0]*100

# %%
filtered_df = filtered_df.drop(columns = ['id', 'pickup_datetime', 'dropoff_datetime', 'pickup_date'])

# %%
# List of all datetime-related features
datetime_columns = ['pickup_day', 'pickup_month', 'pickup_hour', 'pickup_weekday', 'pickup_min']

# Apply Label Encoding to each of the datetime-related columns
for column in datetime_columns:
    filtered_df[column] = enc.fit_transform(filtered_df[column])

# %% [markdown]
# ## 3.5 Data Normalization

# %%
#Predictors and Target Variable
X = filtered_df.drop(['trip_duration'], axis=1)
y = np.log(filtered_df['trip_duration'])

# %%
#Normalizing predictors
from sklearn.preprocessing import StandardScaler
cols = X.columns
ss = StandardScaler()
new_df = ss.fit_transform(X)
new_df = pd.DataFrame(new_df, columns=cols)
new_df.head()

# %% [markdown]
# 🔎 Insights: Normalizing the Dataset using Standard Scaling Technique. Now, why do we use Standard Scaling ? Why not MinMax or Normalizer ?
# 
# - It is because MinMax adjusts the value between 0’s and 1’s , which tend to work better for optimization techniques like Gradient descent and classification algorithms like KNN. Normalizer uses distance measurement like Euclidean or Manhattan, so Normalizer tend to work better with KNN. MinMax Scaling transforms the data to a fixed range, usually between 0 and 1. While this is useful in some machine learning algorithms (especially when dealing with distance-based algorithms like KNN), it is not ideal for PCA. PCA relies on variance, not on the specific range of values. MinMax scaling changes the feature distribution to fall within a certain range but doesn't guarantee that the feature variance is centered or normalized, which is crucial for PCA's covariance matrix computation.
# 
# - Normalizer scales each feature vector to have a unit norm (i.e., making the vector length equal to 1). This is used in distance-based algorithms like KNN or when dealing with sparse data (e.g., text data). It adjusts the magnitude of individual data points but not the spread (variance) of features across the dataset
# 
# - PCA (Principal Component Analysis) is a technique used for dimensionality reduction by finding the directions (principal components) in which the data has the most variance. Standard Scaling (Z-score normalization) is crucial for PCA because of the way PCA operates: PCA computes the covariance matrix of the dataset, which reflects the relationships between features (how each feature varies with others). The next step involves calculating the eigenvalues and eigenvectors of this covariance matrix. The eigenvectors represent the directions (principal components) in which the data varies the most, and the eigenvalues indicate the magnitude of the variance along each of these directions.
# 
# - Standard Scaling (i.e., subtracting the mean and dividing by the standard deviation) ensures that all features have zero mean and unit variance. This is important because PCA is sensitive to the scale of the data. If one feature has much larger values than another, the PCA results will be dominated by that feature, and it will distort the analysis. 

# %% [markdown]
# # 4. Feature Extration: Principal Component Analysis 
# 
# PCA is both a Dimensionality Reduction and Feature Extraction technique. It transforms the original features into new, independent features called principal components. These new features are uncorrelated with each other, meaning that PCA not only reduces the number of dimensions but also eliminates correlations between the variables. So, it's more than just a dimensionality reduction process—it's a way to extract meaningful, independent features from the original data.

# %%
X = new_df

# %%
#Applying PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=len(filtered_df.columns)-1)
pca.fit_transform(X)
var_rat = pca.explained_variance_ratio_
var_rat

# %%
#Variance Ratio vs PC plot
plt.figure(figsize=(15,6))
plt.bar(np.arange(pca.n_components_), pca.explained_variance_, color="grey")

# %%
#Cumulative Variance Ratio

plt.figure(figsize=(10,6))
plt.plot(np.cumsum(var_rat)*100, color="g", marker='o')
plt.xlabel("Principal Components")
plt.ylabel("Cumulative Variance Ratio")
plt.title('Elbow Plot')

# %% [markdown]
# 🔎 Insights: We can consider 12 as a required number of components and extracted new features by transforming the Data.

# %%
#Applying PCA as per required components
pca = PCA(n_components=12)
transform = pca.fit_transform(X)
pca.explained_variance_

# %%
#importance of features in Particular Principal Component
plt.figure(figsize=(25,6))
sns.heatmap(pca.components_, annot=True, cmap="winter")
plt.ylabel("Components")
plt.xlabel("Features")
plt.xticks(np.arange(len(X.columns)), X.columns, rotation=65)
plt.title('Contribution of a Particular feature to our Principal Components')

# %% [markdown]
# # 5. Data Splitting - And Choosing Models

# %% [markdown]
# 🔎 Insights: Let’s pass the PCA Transformed data in our ML Regression Algorithms. To begin with , Linear Regression is a good approach, by splitting our Data into Training and Testing (30%).
# 
# Why Linear Regression , Decision Tree and Random Forest ?
# 
# 1. Linear regression:
# 
# - Simple to explain.
# 
# 
# - Model training and prediction are fast.
# 
# 
# - No tuning is required except regularization.
# 
# 2. Decision Tree:
# 
# - Decision trees are very intuitive and easy to explain.
# 
# 
# - Decision trees are a common-sense technique to find the best solutions to problems with uncertainty.
# 
# 3. Random Forest:
# 
# - It is one of the most accurate learning algorithms available.
# 
# 
# - Random Forest consists of multiple Decision Trees. Each tree makes its own prediction, and the final result is determined by aggregating the predictions from all the trees. This process is called bagging (Bootstrap Aggregating), where each tree is trained on a random subset of the data with replacement. The final prediction is typically based on a majority vote (for classification) or average (for regression) of all the trees' predictions. This approach helps reduce overfitting and improves the model's generalization.
# 
# 
# - Random forests overcome several problems with decision trees like Reduction in overfitting.
# 
# 
# So, I want to approach from base model built using basic Linear Regression and then bring in more Sophisticated Algorithms - Decision Tree & Random Forest. It will give us good idea how Linear Regression performs against Decision Tree Regressor and Random Forest Regressor. Later, we will also approach with same algorithms on "without PCA" data. Finally, we'll evaluate both approaches we took and lay down recommended approach and algorithms.

# %%
#Passing in Transformed values as Predcitors
X = transform
y = np.log(filtered_df['trip_duration']).values

# %%
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_log_error , mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=10)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# %% [markdown]
# ## 5.1. Linear Regression

# %%
#implementing Linear regression

from sklearn.linear_model import LinearRegression
est_lr = LinearRegression()
est_lr.fit(X_train, y_train)
lr_pred = est_lr.predict(X_test)
lr_pred

# %%
#coeficients & intercept
est_lr.intercept_, est_lr.coef_

# %%
#examining scores

print ("Training Score : " , est_lr.score(X_train, y_train))
print ("Validation Score : ", est_lr.score(X_test, y_test))
print ("Cross Validation Score : " , cross_val_score(est_lr, X_train, y_train, cv=5).mean())
print ("R2_Score : ", r2_score(lr_pred, y_test))
#print ("RMSLE : ", np.sqrt(mean_squared_log_error(lr_pred, y_test)))

# %%
#prediction vs real data

plt.figure(figsize=(15,8))
plt.subplot(1,1,1)
sns.distplot(y_test, kde=False, color="black", label="Test")

plt.subplot(1,1,1)
sns.distplot(lr_pred, kde=False, color="g", label="Prediction")
plt.legend()
plt.title("Test vs. Prediction")

# %% [markdown]
# Linear Regression isn't performing well in this data - maybe the relationship undermining is too complex for such a simple model.

# %% [markdown]
# ## 5.2. Decision Trees

# %%
#implementation of decision tree

from sklearn.tree import DecisionTreeRegressor

est_dt = DecisionTreeRegressor(criterion="squared_error", max_depth=10)
est_dt.fit(X_train, y_train)
dt_pred = est_dt.predict(X_test)
dt_pred

# %%
# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, dt_pred)

# Display the result
print("Mean Squared Error (MSE):", mse)

# %%
#examining metrics

print ("Training Score : " , est_dt.score(X_train, y_train))
print ("Validation Score : ", est_dt.score(X_test, y_test))
print ("Cross Validation Score : " , cross_val_score(est_dt, X_train, y_train, cv=5).mean())
print ("R2_Score : ", r2_score(dt_pred, y_test))
print ("RMSLE : ", np.sqrt(mean_squared_log_error(dt_pred, y_test)))

# %%
#prediction vs real data

plt.figure(figsize=(15,8))
plt.subplot(1,1,1)
sns.distplot(y_test, kde=False, color="black", label="Test")

plt.subplot(1,1,1)
sns.distplot(dt_pred, kde=False, color="cyan", label="Prediction")
plt.legend()
plt.title("Test VS Prediction")

# %% [markdown]
# From the above Viz, we can clearly identify that the Decision Tree Algorithm is performing well. It has a high F1 score and a low MSE. The actual data (in grey) and predicted values (in red) are as close as possible. We can conclude that Decision Tree could be a good choice for Trip duration prediction.

# %% [markdown]
# ## 5.3. Random Forest

# %%
#random forest implementation
from sklearn.ensemble import RandomForestRegressor

est_rf = RandomForestRegressor(criterion="squared_error", n_estimators=5, max_depth=10)
est_rf.fit(X_train, y_train)
rf_pred = est_rf.predict(X_test)
rf_pred

# %%
# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, rf_pred)

# Display the result
print("Mean Squared Error (MSE):", mse)

# %%
#examining metrics
print ("Training Score : " , est_rf.score(X_train, y_train))

print ("Validation Score : ", est_rf.score(X_test, y_test))

print ("Cross Validation Score : " , cross_val_score(est_rf, X_train, y_train, cv=5).mean())

print ("R2_Score : ", r2_score(rf_pred, y_test))

print ("RMSLE : ", np.sqrt(mean_squared_log_error(rf_pred, y_test)))

# %%
#prediction vs real data

plt.figure(figsize=(15,8))
plt.subplot(1,1,1)
sns.distplot(y_test, kde=False, color="black", label="Test")

plt.subplot(1,1,1)
sns.distplot(rf_pred, kde=False, color="indigo", label="Prediction")
plt.legend()
plt.title("Test VS Prediction")

# %% [markdown]
# Random Forest also performs pretty well, with a slightly higher Training Score than Decision Trees and slightly lower MSE. I believe hyperparameter tuning would be beneficial.

# %% [markdown]
# ## 5.4. Model Evaluation

# %%
plt.figure(figsize=(10,7))
r2 = pd.DataFrame({'Scores':np.array([r2_score(lr_pred, y_test), r2_score(dt_pred, y_test), r2_score(rf_pred, y_test)]), 'Model':np.array(['Linear Regression', 'Decison Tree', 'Random Forest'])})
r2.set_index('Model').plot(kind="bar", color="brown")
plt.axhline(y=0, color='g')
plt.title("R2 Scores")

# %% [markdown]
# Decision Tree and Random Forest perform much better than Linear Regression. 

# %%
#RMSLE plot
plt.figure(figsize=(10,10))
r2 = pd.DataFrame({'RMSLE':np.array([np.sqrt(mean_squared_log_error(dt_pred, y_test)), np.sqrt(mean_squared_log_error(rf_pred, y_test))]), 'Model':np.array(['Decison Tree', 'Random Forest'])})
r2.set_index('Model').plot(kind="bar", color="lightblue", legend=False)
plt.title("RMSLE - Lesser is Better")

# %%
#Null RMSLE implementation
y_null = np.zeros_like(y_test, dtype=float)
y_null.fill(y_test.mean())
print ("Null RMSLE : ", np.sqrt(mean_squared_log_error(y_test, y_null)))

# %% [markdown]
# Our Decision Tree and Random Forest's RMSLE are much better than the null RMSLE (guessing the mean/random guessing). Although DT risks overfitting the training datast, Random Forest reduces this problem.

# %% [markdown]
# # 6. Fitting the Test Dataset

# %%
test = pd.read_csv("test.csv")

# %%
test_data = test

# %%
test_data.head()

# %%
test_data.pickup_datetime = pd.to_datetime(test_data.pickup_datetime)
test_data['pickup_day'] = test_data['pickup_datetime'].dt.day
test_data['pickup_month'] = test_data['pickup_datetime'].dt.month
test_data['pickup_date'] = test_data['pickup_datetime'].dt.date
test_data['pickup_hour'] = test_data['pickup_datetime'].dt.hour
test_data['pickup_min'] = test_data['pickup_datetime'].dt.minute
test_data['pickup_weekday'] = test_data['pickup_datetime'].dt.weekday

# %%
test_data = test_data.drop(columns = ['id', 'pickup_datetime', 'pickup_date'])

# %%
test_data['store_and_fwd_flag'] = enc.fit_transform(test_data['store_and_fwd_flag'])
test_data['vendor_id'] = enc.fit_transform(test_data['vendor_id'])

# %%
datetime_columns = ['pickup_day', 'pickup_month', 'pickup_hour', 'pickup_weekday', 'pickup_min']

# Apply Label Encoding to each of the datetime-related columns
for column in datetime_columns:
    test_data[column] = enc.fit_transform(test_data[column])

# %%
pca = PCA(n_components=12)
X_test = pca.fit_transform(test_data)
pca.explained_variance_

# %%
# Make predictions for test data
est_rf.fit(X, y)
dt_pred = est_dt.predict(X_test)
rf_pred = est_rf.predict(X_test)

# %%
# Create the submission DataFrame
submission = pd.DataFrame({'id': test['id'], 'trip_duration': rf_pred})
submission['trip_duration'] = np.expm1(submission['trip_duration'])

# Save the predictions to a CSV for submission
submission.to_csv("submission.csv", index=False)

# %%
submission.shape[0]


