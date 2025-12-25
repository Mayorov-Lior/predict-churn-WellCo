import random
import numpy as np
import pandas as pd
from preprocess_data import PreprocessData
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report

class PredictChurn:
    def __init__(self, check_validation:bool=True):
        self.check_validation = check_validation
        if self.check_validation:
            self.df_usage, self.df_visits, self.df_claims, self.df_churn_labels = self.read_data(train=True)
            self.split_data()
        else:
            self.df_usage_train, self.df_visits_train, self.df_claims_train, self.df_churn_labels_train = self.read_data(train=True)
            self.df_usage_test, self.df_visits_test, self.df_claims_test, self.df_churn_labels_test = self.read_data(train=False)
        
        self.df_test = None
        self.preprocess_data()    

    def read_data(self, train:bool=True):
        folder = 'train/' if train else 'test/test_'
        df_usage = pd.read_csv(f'{folder}app_usage.csv') # pd.read_csv('test/test_app_usage.csv') 
        df_visits = pd.read_csv(f'{folder}web_visits.csv')
        df_claims = pd.read_csv(f'{folder}claims.csv')
        df_churn_labels = pd.read_csv(f'{folder}churn_labels.csv')
        return df_usage, df_visits, df_claims, df_churn_labels

    def split_data(self):
        # split 90% train, 10% test (validation) by member_id )
        random.seed(42)
        members = list(set(self.df_visits['member_id'].unique().tolist() + self.df_claims['member_id'].unique().tolist() + self.df_usage['member_id'].unique().tolist()))
        train_size = int(0.9 * len(members))
        train_members = random.sample(members, train_size)

        def split_dataset(df, train_members):
            df_train = df[df['member_id'].isin(train_members)]
            df_test = df[~df['member_id'].isin(train_members)]
            return df_train, df_test

        self.df_visits_train, self.df_visits_test = split_dataset(self.df_visits, train_members)
        self.df_usage_train, self.df_usage_test = split_dataset(self.df_usage, train_members)
        self.df_claims_train, self.df_claims_test = split_dataset(self.df_claims, train_members)
        self.df_churn_labels_train, self.df_churn_labels_test = split_dataset(self.df_churn_labels, train_members)

        print(self.df_churn_labels_train['churn'].value_counts())
        print(self.df_churn_labels_test['churn'].value_counts())

    def preprocess_data(self):
        # preprocess train and test data
        train_preprocessor = PreprocessData(df_usage=self.df_usage_train, df_visits=self.df_visits_train, df_claims=self.df_claims_train, df_churn_labels=self.df_churn_labels_train)
        df_train = train_preprocessor.join_data()

        test_preprocessor = PreprocessData(df_usage=self.df_usage_test, df_visits=self.df_visits_test, df_claims=self.df_claims_test, df_churn_labels=self.df_churn_labels_test)
        self.df_test = test_preprocessor.join_data()

        df_train.drop(columns=['outreach','signup_date'], inplace=True) # if had more time, would have engineered features from these

        self.X_train, self.y_train = df_train.drop(columns=['member_id', 'churn']), df_train['churn']
        self.X_test, self.y_test = self.df_test.drop(columns=['member_id', 'churn', 'outreach','signup_date']), self.df_test['churn']

    def train_model_and_predict(self, th:float=0.5, top_percentage:float=1):
        # train XGBoost model, predict on test set and return results
        scale_pos_weight = len(self.y_train[self.y_train==0]) / len(self.y_train[self.y_train==1])
        model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight,max_depth=1, learning_rate=0.2, n_estimators=50, objective='binary:logistic') # 0.66

        model.fit(self.X_train, self.y_train)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]

        y_train_pred_proba = model.predict_proba(self.X_train)[:, 1]
        auc_train = roc_auc_score(self.y_train, y_train_pred_proba)
        if self.check_validation:
            print(f"Train AUC: {auc_train:.3f}")
            print(classification_report(self.y_train, (y_train_pred_proba > 0.5).astype(int)))

        auc = roc_auc_score(self.y_test, y_pred_proba)
        classification_report_ = classification_report(self.y_test, (y_pred_proba > th).astype(int))
        if not self.check_validation:
            with open('classification_report_test.txt', 'w') as f:
                f.write(classification_report_)
            with open('auc_test.txt', 'w') as f:
                f.write(f"ROC-AUC_test={auc:.3f}")
        else:
            print(f"Test AUC: {auc}")
            print(classification_report_)

        df_results_train = pd.DataFrame({'prioritization_score':y_train_pred_proba, 'true':self.y_train}).sort_values(by='prioritization_score', ascending=False)
        df_results_test = pd.DataFrame({'member_id':self.df_test['member_id'], 'prioritization_score':y_pred_proba, 'true':self.y_test,
                                        'outreach':self.df_test['outreach'], 'signup_date':self.df_test['signup_date']
                                        }).sort_values(by='prioritization_score', ascending=False).reset_index(drop=True)
        df_results_test['rank'] = np.arange(1, len(df_results_test)+1)
        
        if not self.check_validation:
            df_results_test = df_results_test.head(int(top_percentage * len(df_results_test)))
            df_results_test.to_csv('churn_test_predictions.csv', index=False)

        return df_results_train, df_results_test