import numpy as np
import pandas as pd
import os
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split


def load_data(feature_dir, label_dir, flow_case, feature_fields, label_fields):
    feature_data = []
    label_data = []

    for field in feature_fields:
        file_path = os.path.join(feature_dir, f"{flow_case}_{field}.npy")
        feature_data.append(np.load(file_path).flatten())

    for field in label_fields:
        file_path = os.path.join(label_dir, f"{flow_case}_{field}.npy")
        label_data.append(np.load(file_path).flatten())
    
    features_df = pd.DataFrame(np.array(feature_data).T, columns=feature_fields)
    labels_df = pd.DataFrame(np.array(label_data).T, columns=label_fields)
    return features_df, labels_df


feature_dir = '../data/features' 
label_dir = '../data/labels'
feature_fields = ['Ux', 'Uy', 'Uz', 'p', 'k', 'epsilon']
label_fields = ['um', 'vm', 'wm', 'uu', 'vv', 'ww']
flow_case = 'DUCT_1100' 


X, y = load_data(feature_dir, label_dir, flow_case, feature_fields, label_fields)


data = pd.concat([X, y], axis=1)
target_columns = y.columns.tolist()


train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)


predictors = {}
for target in target_columns:
    print(f"Training model for target: {target}")
    predictors[target] = TabularPredictor(label=target, eval_metric='mean_squared_error').fit(
        train_data=train_data, time_limit=1800, presets='best_quality'
    )


for target in target_columns:
    print(f"Evaluating model for target: {target}")
    performance = predictors[target].evaluate(test_data)
    print(f"Performance for {target}:", performance)


for target in target_columns:
    save_path = predictors[target].save()
    print(f"Model for {target} saved at: {save_path}")