import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD

class ProcessData:
    def __init__(self, df_visits: pd.DataFrame, df_usage: pd.DataFrame):
        self.df_visits = df_visits
        self.df_usage = df_usage
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.fitted_svd = None


    def process_visits(self) -> pd.DataFrame:
        # embeddings for 'title' + 'description' using SentenceTransformer
        self.df_visits['text_to_embed'] = self.df_visits['title'].fillna('') + ' ' + self.df_visits['description'].fillna('')
        self.embedd_column(col='text_to_embed')

        # perform dummies on title column
        dummies = pd.get_dummies(self.df_visits['title'], prefix='page')
        self.df_visits = pd.concat([self.df_visits.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)

        self.df_visits.drop(columns = ['description','text_to_embed','url'], inplace=True)
        self.aggregate_visits()

    import scipy.stats as stats

    def gap_stats(self, group):
        if len(group) < 2:
            return pd.Series([np.nan]*6, index=['gap_mean', 'gap_median', 'gap_min', 'gap_max', 'gap_skew', 'gap_kurt'])
        
        group = pd.to_datetime(group)
        diffs = group.sort_values().diff().dropna().dt.total_seconds()
        return pd.Series({
            'gap_mean': diffs.mean(),
            'gap_median': diffs.median(),
            'gap_min': diffs.min(),
            'gap_max': diffs.max(),
            'gap_skew': diffs.skew(),
            'gap_kurt': diffs.kurtosis()
        })

    def aggregate_visits(self) -> pd.DataFrame:
        agg_dict_emb = {col: ['mean','std'] for col in self.df_visits.columns if col.startswith('emb_')}
        agg_dict_page = {col: 'sum' for col in self.df_visits.columns if col.startswith('page')}

        agg_dict = {**agg_dict_emb, **agg_dict_page, 'title': ['count', 'nunique']}

        gap_df = self.df_visits.groupby('member_id')['timestamp'].apply(self.gap_stats).unstack().reset_index()
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
        embedding_df = pd.DataFrame(embeddings, columns=[f'emb_{i}' for i in range(embeddings.shape[1])])
        self.df_visits = pd.concat([self.df_visits.reset_index(drop=True), embedding_df], axis=1)
