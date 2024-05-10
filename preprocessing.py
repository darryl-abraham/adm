import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import MinCovDet
from scipy.stats import chi2
from sklearn.impute import KNNImputer
import time
import sys


class Preprocessor:
    def __init__(self, path_to_data):
        self.df = pd.read_csv(path_to_data)
        self.numerical_columns = self.df.drop(columns=['Unnamed: 0', 'time_signature', 'mode', 'key']).select_dtypes(include=['float64', 'int64']).columns
        self.categorical_columns = list(list(set(self.df.columns.to_list()) - set(self.df[self.numerical_columns].columns.tolist()) | {'Unnamed: 0', 'time_signature', 'mode', 'key'}))
        self.df_num = self.df[self.numerical_columns]
        self.df_cat = self.df[self.categorical_columns]

    def update(self):
        self.numerical_columns = self.df.drop(columns=['Unnamed: 0', 'time_signature', 'mode', 'key']).select_dtypes(include=['float64', 'int64']).columns
        self.categorical_columns = list(list(set(self.df.columns.to_list()) - set(self.df[self.numerical_columns].columns.tolist()) | {'Unnamed: 0', 'time_signature', 'mode', 'key'}))
        self.df_num = self.df[self.numerical_columns]
        self.df_cat = self.df[self.categorical_columns]

    def get_uni_outliers(self, remove=False, impute=False, save=False):
        self.update()
        uni_outliers = {}
        for col in self.df_num.columns:
            q1 = self.df_num[col].quantile(0.25)
            q3 = self.df_num[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            uni_outliers[col] = self.df_num[(self.df_num[col] < lower_bound) | (self.df_num[col] > upper_bound)].index
        if remove:
            for col in uni_outliers.keys():
                self.df.loc[uni_outliers[col], col] = np.nan
            if impute:
                self.knn_impute()
        if save:
            self.save()
        return uni_outliers

    def get_multi_outliers(self, remove=False, save=False):
        self.update()
        multi_outliers_dict = {}
        multi_outliers_list = []
        for genre in self.df['track_genre'].unique():
            genre_df = self.df[self.df['track_genre'] == genre]
            genre_df_num = genre_df.select_dtypes(include=['float64', 'int64'])
            genre_df_num = genre_df_num.drop(columns=['time_signature', 'mode', 'key'])
            robust_cov = MinCovDet(support_fraction=1).fit(genre_df_num)
            mahalanobis = robust_cov.mahalanobis(genre_df_num)
            threshold = chi2.ppf(0.99, df=genre_df_num.shape[1])
            multi_outliers_dict[genre] = np.where(mahalanobis > threshold)
            multi_outliers_list.extend(list(genre_df.iloc[multi_outliers_dict[genre][0]]['Unnamed: 0']))
        if remove:
            self.drop(index=multi_outliers_list)
        if save:
            self.save()
        return multi_outliers_dict

    def knn_impute(self, save=False):
        self.update()
        imputer = KNNImputer(n_neighbors=5, missing_values=np.nan)

        imputed_df = pd.DataFrame(imputer.fit_transform(self.df_num), columns=self.df_num.columns)

        self.df = pd.concat([imputed_df, self.df_cat], axis=1)

        if save:
            self.save()

        return self.df

    def scale_data(self, save=False):
        scaler = StandardScaler()
        scaled_numerical = scaler.fit_transform(self.df_num)
        scaled_df = pd.DataFrame(scaled_numerical, columns=self.numerical_columns)

        self.df = pd.concat([scaled_df, self.df_cat], axis=1)

        if save:
            self.save()

        return self.df

    def drop(self, index):
        self.df.drop(index, inplace=True)

    def save(self):
        self.df.to_csv('data/preprocessed_data.csv', index=False)
        print('Data saved to data/preprocessed_data.csv')

    def preprocess(self, scale=True, handle_uni_outliers=True, remove_multi_outliers=True, save=True):
        print('Preprocessing data...')
        start = time.time()

        if scale:
            print('Scaling data...')
            start_time = time.time()
            self.scale_data()
            end_time = time.time()
            print(f'Scaling data done in {end_time-start_time} seconds')

        if handle_uni_outliers:
            print('Replacing univariate outliers by NaN...')
            start_time = time.time()
            self.get_uni_outliers(remove=True, impute=True)
            print('Imputing univariate outliers by K-NN...')
            self.knn_impute()
            end_time = time.time()
            print(f'Univariate outlier handling done in {end_time-start_time} seconds')

        if remove_multi_outliers:
            print('Removing multivariate outliers...')
            start_time = time.time()
            self.get_multi_outliers(remove=True)
            end_time = time.time()
            print(f'Removing multi outliers done in {end_time - start_time} seconds')

        end_time = time.time()
        print('Preprocessing done in', end_time - start, 'seconds')

        self.df = self.df.drop(columns=['Unnamed: 0'])

        if save:
            self.save()


if __name__ == '__main__':
    preprocessor = Preprocessor('data/spotifydata.csv')
    preprocessor.preprocess(True, True, True, True)
