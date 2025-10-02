### Salary-prediction ###
Predict the salaries for the job postings contained in test_features.csv (__1 million rows__). Train and predict salary with linear regression and light gradient boosting machine. Interpret model results.<br>output CSV files with feature importance.
## Data.zip ##
1. train_features.csv : Each row represents metadata for an individual job. The “jobId” column represents a unique identifier for the job posting. The remaining columns describe features of the job.
2. train_salaries.csv : Each row associates a “jobId” with a “salary”.</li>
3. test_features.csv : Similar to train_features.csv , each row represents metadata for an individual job.
## Model accuracy ##
interprets model results
## predict_salary.py ##
main model training function
## Process ##
1. download the repository
2. extract data.zip, move train_features.csv and train_salaries.csv to main project
3. uncomment optimize(data,50) to fine tune lgbm hyperparameters
4. update linear_train(...,True), lgb_train(...,True) to output feature importance csv
