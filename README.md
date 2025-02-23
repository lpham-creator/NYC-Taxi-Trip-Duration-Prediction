# NYC Taxi Trip Duration Prediction

## Introduction
This project aims to predict taxi trip duration using a dataset from NYC Taxi and Limousine Commission (TLC). The goal is to build a robust machine-learning model that accurately estimates trip durations based on various features such as pickup/dropoff locations, timestamps, vendor details, and trip metadata.

## Data Source
The dataset comes from the **Kaggle NYC Taxi Trip Duration competition** and includes:
- **Pickup & Dropoff Datetime**
- **Passenger Count**
- **Trip Distance & Duration**
- **Vendor Information**
- **Store & Forward Flag** (whether data was stored before transmission)

## Approach & Key Steps
1. **Exploratory Data Analysis (EDA)**  
   - Analyzed passenger counts, trip duration distributions, and vendor-wise trip statistics.  
   - Identified outliers such as **trips longer than 10 hours and shorter than 5 minutes**.

2. **Feature Engineering**  
   - Extracted **time-based features** (day, month, hour, minute, weekday) from datetime columns.  
   - Encoded categorical variables like `store_and_fwd_flag` and `vendor_id` using Label Encoding.  
   - Removed irrelevant or unreliable data points (e.g., trips with 0 passengers, extreme outliers).

3. **Dimensionality Reduction with PCA**  
   - Applied **Principal Component Analysis (PCA)** to reduce feature dimensionality while preserving variance.  
   - Ensured PCA was applied to both training and test sets to maintain consistency.

4. **Model Selection & Training**  
   - Implemented **Decision Trees & Random Forest** for regression modeling.  
   - Evaluated models using **RMSLE (Root Mean Squared Logarithmic Error)** to handle skewed trip durations.

5. **Prediction & Submission**  
   - Ensured the final predictions were **exponentiated** (`np.expm1()`) to reverse the log transformation.  
   - Saved the predictions in **submission.csv** for competition submission.

## ⚠Key Struggles & Challenges
- **Log-Transformed Target:** Initially, trip durations were predicted in log scale, requiring careful transformation (`exp()` or `expm1()`).
- **Feature Selection:** Managing high-dimensional data without overfitting, leading to the decision to use PCA.
- **Model Interpretability:** Balancing model complexity (Random Forest vs. Decision Trees) while keeping training time reasonable.
- **Handling Missing or Incorrect Data:** Filtering out extreme outliers (e.g., unrealistic trip durations and passenger counts).

## Insights & Learnings
✔ **Trip Duration Distribution is Skewed**  
  Most trips have relatively short durations, with a significant number of trips under 1 hour. This indicates a heavy **right skew**, where a few long trips have a disproportionate impact on the model. Handling this with **log transformation** (RMSLE) helped in improving the model’s accuracy.

✔ **Passenger Count Patterns**  
  **One-passenger trips** dominated the dataset, while **trips with 7+ passengers** were rare and considered outliers. These outliers, which only represented a small fraction of the data, were removed to reduce noise in the predictions.

✔ **Date and Time-Based Features Are Crucial**  
  **Time-based features** (hour, weekday, etc.) showed significant influence on trip duration. **Peak hours** such as rush hour (morning and evening) were correlated with longer durations. Therefore, extracting **hour**, **minute**, **day of the week**, and **month** from the datetime columns was essential.

✔ **Removing Outliers Leads To Better Prediction**  
  **Outliers** can significantly distort model predictions by introducing noise and skewing relationships between variables. By identifying and removing extreme values, the model can learn patterns more effectively, leading to improved accuracy and generalization. This preprocessing step enhances the model’s ability to make reliable predictions on unseen data.


## 🚀 Next Steps & Improvements
- **Incorporate Clustering (e.g., K-Means)** to group similar pickup/dropoff locations.
- **Try KNN-Based Features** to estimate trip durations based on nearby historical trips.
- **Experiment with Gradient Boosting (e.g., XGBoost, LGBM)** for potential performance gains.
- **Improve feature selection** using mutual information or permutation importance.

