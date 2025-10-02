import pandas as pd
import lightgbm as lgb
import shap, optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression

def load_data():
    """
    load salary data
    Returns:
        data: 2D DataFrame
    """
    print("load data...")
    train_features = pd.read_csv("train_features.csv")
    train_salaries = pd.read_csv("train_salaries.csv")
    # Merge labels into features for training
    data = train_features.merge(train_salaries, on="jobId")    
    return data

def clean_data(data):
    """
    clean salary data

    Parameters
    -------
    data: 2D DataFrame

    Returns
    -------
    data: 2D DataFrame
    categoric_cols: list[str]
    numeric_cols: list[str]
    """
    print("clean data...")
    # Drop rows with missing or invalid salary
    data = data[data.loc[:,"salary"].notnull() & (data.loc[:,"salary"] > 0)]
    
    # Drop rows with missing values
    categoric_cols = ["jobType", "degree", "major", "industry","companyId"]
    numeric_cols=["yearsExperience","milesFromMetropolis"]
    data=data.dropna(subset=categoric_cols)
    data = data.dropna(subset=numeric_cols)    
    data = data[
        (data["yearsExperience"] >= 0) &
        (data["yearsExperience"] < 100) &
        (data["milesFromMetropolis"] > 0) &
        (data["milesFromMetropolis"] < 3000)
    ]    
    data = data.reset_index(drop=True)    
    # Optimize data types for categorical columns
    for col in categoric_cols:        
        data[col] = data[col].astype('category')
    return data, categoric_cols, numeric_cols

def linear_train(data,categoric_cols,numeric_cols,eval):
    """train salary data with linear regression, add dummy variables for categorical features

    Args:
        data (2D DataFrame): clean salary data
        categoric_cols (list[str]): categorical features
        numeric_cols (list[str]): numerical features
        eval (_bool_): if true, calcuates feature importance, output results to csv
    """
    print("training linear regression...")
    X = data.drop(columns=["jobId", "salary"])
    y = data["salary"]

    # One-hot encode categoricals
    X = pd.get_dummies(X,columns=categoric_cols)

    #Train/Test Split    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr = LinearRegression()
    #train model
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_val)

    rmse_lr = root_mean_squared_error(y_val, y_pred_lr)
    print(f"Linear Regression RMSE: {rmse_lr:.2f}")
    
    #Feature importance
    if eval:
        coef_df = pd.DataFrame({
            "feature": X_train.columns,
            "coefficient": lr.coef_})
        feature_groups = {
            "jobType": [col for col in X_train.columns if col.startswith("jobType_")],
            "degree": [col for col in X_train.columns if col.startswith("degree_")],
            "major": [col for col in X_train.columns if col.startswith("major_")],
            "industry": [col for col in X_train.columns if col.startswith("industry_")],
            "companyId": [col for col in X_train.columns if col.startswith("companyId_")]
        }
        sorted_blocks = []    
        # Sort each categorical group descending
        for _, cols in feature_groups.items():
            block = coef_df[coef_df["feature"].isin(cols)].sort_values(
                by="coefficient", ascending=False
            )
            sorted_blocks.append(block)
        # Add numeric features at the end
        numeric_block = coef_df[coef_df["feature"].isin(numeric_cols)].sort_values(
            by="coefficient", ascending=False
        )
        sorted_blocks.append(numeric_block)

        file="linear_regression_feature_effects.csv"
        pd.concat(sorted_blocks).to_csv(file,index=False)
        print(f"saved Features (Linear regression) to {file}")


def lgb_train(data,categoric_cols,numeric_cols,eval):
    """train salary data with LightGBM, Light Gradient Boosting Machine.
    details: https://github.com/microsoft/LightGBM
    Args:
        data (2D DataFrame): clean salary data
        categoric_cols (list[str]): categorical features
        numeric_cols (list[str]): numerical features
        eval (bool): if true, calculates feature importance, output results to csv
    """
    print("lgb training...")
    X = data.drop(columns=["jobId", "salary"])
    y = data["salary"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categoric_cols)
    lgb_val   = lgb.Dataset(X_val, y_val, reference=lgb_train, categorical_feature=categoric_cols)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 32,
        "learning_rate": 0.05,
        "feature_fraction": 0.5,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "early_stopping_round":10
    }
    #train model 
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        num_boost_round=200       
    )    
    y_pred_lgb = model.predict(X_val, num_iteration=model.best_iteration)

    #evaluate model
    rmse_lgb = root_mean_squared_error(y_val, y_pred_lgb)
    print(f"LightGBM RMSE: {rmse_lgb:.2f}")
    
    #Feature Importance
    if eval:
        X_val_sample = X_val.sample(5000, random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val_sample)
        categoric_effects=[]
        numeric_effects = []
        
        for feature in numeric_cols:
            n_index = list(X_val.columns).index(feature)
            n_shap = shap_values[:, n_index]
            numeric_effects.append({
                "feature": feature,
                "shap_value": n_shap.mean()})        
        df_numeric_effects = pd.DataFrame(numeric_effects)

        for feature in categoric_cols:
            c_index = list(X_val.columns).index(feature)
            c_shap = shap_values[:, c_index]
            df_cat = pd.DataFrame({
                "feature": X_val_sample[feature].values,
                "shap_value": c_shap})
            cat = df_cat.groupby("feature",observed=True,as_index=False)["shap_value"].mean().sort_values(by="shap_value",ascending=False)        
            cat = cat[["feature", "shap_value"]]
            categoric_effects.append(cat)
        file="lgbm_feature_effects.csv"
        
        pd.concat(categoric_effects+[df_numeric_effects]).to_csv(file,index=False)
        print(f"Saved Features to (LGBM) {file}")
    
def objective(trial,data):
    """fine tune model hyperparameters efficently

    Args:
        trial (function): model hyperparameters
        data (2D DataFrame): clean salary data

    Returns:
        float: RMSE
    """
    # Suggest hyperparameters
    param = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "max_depth": trial.suggest_int("max_depth", -1, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "early_stopping_rounds":trial.suggest_int("early_stopping_rounds",1,50)
    }    
    
    X = data.drop(columns=["jobId","salary"])
    y = data["salary"]    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categoric_cols)
    lgb_val   = lgb.Dataset(X_val, y_val, reference=lgb_train, categorical_feature=categoric_cols)
    model = lgb.train(
        param,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        num_boost_round=200
    )

    # Predict on validation set
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    rmse = root_mean_squared_error(y_val, y_pred)
    return rmse

def optimize(data,n_trials):
    """implement fune tune model

    Args:
        data (2D DataFrame): clean salary data
        n_trials (int): number of trials
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial:objective(trial,data), n_trials)
    print("Best RMSE:", study.best_value)
    print("Best params:", study.best_params)

if __name__ == "__main__":
    data=load_data()
    data,categoric_cols,numeric_cols=clean_data(data)    
    #linear regression
    linear_train(data,categoric_cols,numeric_cols,False)
    #light gradient boosting
    lgb_train(data,categoric_cols,numeric_cols,False)
    #optimize(data,50)