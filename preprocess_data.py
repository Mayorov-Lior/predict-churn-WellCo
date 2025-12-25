import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
import scipy.stats as stats

class PreprocessData:
    def __init__(self, df_visits: pd.DataFrame, df_usage: pd.DataFrame, df_claims: pd.DataFrame, 
                         df_churn_labels: pd.DataFrame, model: SentenceTransformer = None):
        self.df_visits = df_visits
        self.df_usage = df_usage
        self.df_claims = df_claims
        self.df_churn_labels = df_churn_labels
        if model is None:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.model = model
        self.fitted_svd = None

    def join_data(self) -> pd.DataFrame:
        self.process_claims()
        self.process_usage()
        self.process_visits()

        df_merged = self.df_visits.merge(self.df_usage, on='member_id', how='outer')
        df_merged = df_merged.merge(self.df_claims, on='member_id', how='outer')
        df_merged = df_merged.merge(self.df_churn_labels, on='member_id', how='outer')

        df_merged.fillna(0, inplace=True) # would have handled differently with more time
        df_merged = df_merged.sample(frac=1).reset_index(drop=True)

        return df_merged

    def process_claims(self) -> None:
        dummies = pd.get_dummies(self.df_claims['icd_code'], prefix='icd_code')
        self.df_claims = pd.concat([self.df_claims.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
        self.df_claims = self.df_claims.groupby('member_id').agg({col: 'sum' for col in dummies.columns}).reset_index()
        # self.df_claims.drop(columns=['icd_code','diagnosis_date'], inplace=True)

    def process_usage(self) -> None:
        self.df_usage = self.df_usage.groupby('member_id')['timestamp'].apply(lambda x: self.gap_stats(x, prefix='usage')).unstack().reset_index()
        self.df_usage = self.df_usage.drop(columns=['level_1']) if 'level_1' in self.df_usage.columns else self.df_usage

    def process_visits(self) -> None:
        # embeddings for 'title' + 'description' using SentenceTransformer
        self.df_visits['text_to_embed'] = self.df_visits['title'].fillna('') + ' ' + self.df_visits['description'].fillna('')
        self.embedd_column(col='text_to_embed')

        # perform dummies on title column
        dummies = pd.get_dummies(self.df_visits['title'], prefix='page')
        self.df_visits = pd.concat([self.df_visits.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)

        self.df_visits.drop(columns = ['description','text_to_embed','url'], inplace=True)
        self.aggregate_visits()

    def gap_stats(self, group, prefix='visits'):
        if len(group) < 2:
            return pd.Series([0]*6, index=[f'{prefix}_gap_mean', f'{prefix}_gap_median', f'{prefix}_gap_min', f'{prefix}_gap_max', f'{prefix}_gap_skew', f'{prefix}_gap_kurt'])
        
        group = pd.to_datetime(group)
        diffs = group.sort_values().diff().dropna().dt.total_seconds()
        return pd.Series({
            f'{prefix}_gap_mean': diffs.mean(),
            f'{prefix}_gap_median': diffs.median(),
            f'{prefix}_gap_min': diffs.min(),
            f'{prefix}_gap_max': diffs.max(),
            f'{prefix}_gap_skew': diffs.skew(),
            f'{prefix}_gap_kurt': diffs.kurtosis()
        }).fillna(0)

    def aggregate_visits(self) -> pd.DataFrame:
        agg_dict_emb = {col: ['mean','std'] for col in self.df_visits.columns if col.startswith('visit_emb_')}
        agg_dict_page = {col: 'sum' for col in self.df_visits.columns if col.startswith('page')}

        agg_dict = {**agg_dict_emb, **agg_dict_page, 'title': ['count', 'nunique']}

        gap_df = self.df_visits.groupby('member_id')['timestamp'].apply(lambda x: self.gap_stats(x, prefix='visits')).unstack().reset_index()
        gap_df = gap_df.drop(columns=['level_1']) if 'level_1' in gap_df.columns else gap_df

        self.df_visits = self.df_visits.groupby('member_id').agg(agg_dict).reset_index()
        self.df_visits.columns = ['_'.join(col).strip().strip('_') if isinstance(col, tuple) else col for col in self.df_visits.columns]
    
        self.df_visits = self.df_visits.merge(gap_df, on='member_id', how='left')
        
    def embedd_column(self, col='text_to_embed', n_dims=50) -> pd.DataFrame:
        df_unique_texts = self.df_visits[col].fillna('').unique().tolist()
        embeddings = self.model.encode(df_unique_texts)
        if n_dims < embeddings.shape[1]:
            if self.fitted_svd is None: # fit train data 
                svd = TruncatedSVD(n_components=n_dims)
                self.fitted_svd = svd.fit(embeddings)  # will caps at n_components to min(n_samples, n_features)
            embeddings = self.fitted_svd.transform(embeddings)
        embeddings_mapping = dict(zip(df_unique_texts, embeddings))
        embeddings = np.array([embeddings_mapping[text] for text in self.df_visits[col].fillna('')])
        embedding_df = pd.DataFrame(embeddings, columns=[f'visit_emb_{i}' for i in range(embeddings.shape[1])])
        self.df_visits = pd.concat([self.df_visits.reset_index(drop=True), embedding_df], axis=1)
