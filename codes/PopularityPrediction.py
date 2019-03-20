#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pytz
import json
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold


def first_look(hashtag, plot=False):
    filename = 'data/tweets_#{:}.txt'.format(hashtag)
    date = []
    n_followers = []
    citation = []
    with open(filename, 'r') as f:
        for line in f:
            tweet = json.loads(line)
            date.append(tweet['citation_date'])
            n_followers.append(tweet['author']['followers'])
            citation.append(tweet['metrics']['citations']['total'])
    data = pd.DataFrame({'date': date, 'followers': n_followers,
                         'citation': citation})

    # create hour column
    data['time'] = data['date'].apply(datetime.datetime.utcfromtimestamp)
    data['hour'] = data['time'].apply(lambda t: t.replace(
        microsecond=0, second=0, minute=0))
    span = data['hour'].max() - data['hour'].min()
    tweet_per_hour = data.shape[0] / (span.days*24 + span.seconds/3600)
    average_followers = data['followers'].mean()
    average_retweets = data['citation'].mean()

    result = {'hashtag': hashtag,
              'average tweets per hour': tweet_per_hour,
              'average followers per tweet': average_followers,
              'average retweets per tweet': average_retweets}

    if plot:
        tweets_over_time = data.groupby('hour').size()
        plt.plot(tweets_over_time.values, label=hashtag)
        plt.xlabel('hour')
        plt.ylabel('number of tweets')
        plt.legend()
        plt.show()

    return data, result


def extract_Xy(data):
    data_agg = data.groupby('hour', as_index=False).agg(
        {'followers': ['count', 'sum', 'max'],
         'citation': 'sum'})
    data_agg['hour'] = data_agg['hour'].apply(lambda x: x.hour)
    # Using ravel, and a string join to flatten column
    columns_ravel = lambda df: ["_".join(x) for x in df.columns.ravel()]
    data_agg.columns = columns_ravel(data_agg)
    data_agg.rename(columns={'hour_': 'hour_of_day',
                             'followers_count': 'n_tweets',
                             'citation_sum': 'total_retweets'}, inplace=True)
    data_agg['target'] = data_agg['n_tweets'].shift(-1)
    data_agg = data_agg.iloc[:-1]
    X, y = data_agg.drop(columns=['target']), data_agg['target']
    return X, y


def fit_OLS(X, y, hashtag):
    X_c = sm.add_constant(X)
    reg = sm.OLS(y, X_c)
    est = reg.fit()
    summary = [dict(hashtag=hashtag,
                    mse_total=est.mse_total,
                    r_squared=est.rsquared,
                    tvalues=dict(est.tvalues),
                    pvalues=dict(est.pvalues))]
    summary_df = pd.DataFrame(summary)
    # expand tvalues/pvalues into columns
    for column in ['tvalues', 'pvalues']:
        df_expand = pd.DataFrame(summary_df[column].values.tolist())
        df_expand.rename(columns=lambda x: '_'.join([column, x]), inplace=True)
        summary_df = pd.concat([summary_df.drop(columns=column),
                                df_expand], axis=1)
    summary_df = summary_df.T
    summary_df.iloc[1:, :] = summary_df.iloc[1:, :].astype(float).round(4)
    return summary_df


def new_features(hashtag):
    filename = 'data/tweets_#{}.txt'.format(hashtag)
    date = []
    n_followers = []
    citation = []
    with open(filename, 'r') as f:
        for line in f:
            tweet = json.loads(line)
            date.append(tweet['citation_date'])
            n_followers.append(tweet['author']['followers'])
            citation.append(tweet['metrics']['citations']['total'])
    data = pd.DataFrame({'date': date, 'followers': n_followers,
                         'citation': citation})
    # create hour column


    pass


def split_periods(hashtag):
    filename = 'data/tweets_#{}.txt'.format(hashtag)
    date = []
    n_followers = []
    citation = []
    with open(filename, 'r') as f:
        for line in f:
            tweet = json.loads(line)
            date.append(tweet['citation_date'])
            n_followers.append(tweet['author']['followers'])
            citation.append(tweet['metrics']['citations']['total'])
    data = pd.DataFrame({'date': date, 'followers': n_followers,
                         'citation': citation})

    # create hour column
    pst_tz = pytz.timezone('America/Los_Angeles')
    data['time'] = data['date'].apply(
        lambda x: datetime.datetime.fromtimestamp(x, pst_tz))

    before_df = data.loc[
        data['time'] < datetime.datetime(2015, 2, 1, 8, tzinfo=pst_tz)]
    between_df = data.loc[
        (data['time'] >= datetime.datetime(2015, 2, 1, 8, tzinfo=pst_tz))
        & (data['time'] <= datetime.datetime(2015, 2, 1, 20, tzinfo=pst_tz))]
    after_df = data.loc[
        data['time'] > datetime.datetime(2015, 2, 1, 20, tzinfo=pst_tz)]

    before_df['hour'] = before_df['time'].apply(
        lambda t: t.replace(microsecond=0, second=0, minute=0))
    after_df['hour'] = after_df['time'].apply(
        lambda t: t.replace(microsecond=0, second=0, minute=0))
    between_df['hour'] = between_df['time'].apply(
        lambda t: t.replace(microsecond=0, second=0,
                            minute=t.minute-(t.minute % 5)))
    return before_df, between_df, after_df

def main():
    #%% Q1-2: first look
    results = []
    for hashtag in ['gohawks', 'gopatriots', 'patriots', 'sb49']:
        results.append(first_look(hashtag)[1])
    for hashtag in ['nfl', 'superbowl']:
        results.append(first_look(hashtag, plot=True)[1])
    results_df = pd.DataFrame(results)

    #%% Q3: linear regression
    hashtag_list = ['gohawks', 'gopatriots', 'patriots', 'sb49', 'nfl',
                    'superbowl']
    summary = []
    for hashtag in hashtag_list:
        data = first_look(hashtag)[0]
        X, y = extract_Xy(data)
        summary.append(fit_OLS(X, y, hashtag))
    summary_df = pd.concat(summary, axis=1, ignore_index=True)

    #%% Q4-5: design new features

    #%% Q6: split time periods
    hashtag_list = ['gohawks', 'gopatriots', 'patriots', 'sb49', 'nfl',
                    'superbowl']
    summary = []
    for hashtag in hashtag_list:
        before_df, between_df, after_df = split_periods(hashtag)
        for i, data in enumerate([before_df, between_df, after_df]):
            X, y = extract_Xy(data)
            summary.append(
                fit_OLS(X, y, hashtag+'_'+['before', 'between', 'after'][i]))
    summary_df = pd.concat(summary, axis=1, ignore_index=True)
    summary_before = summary_df.loc[:,
                     ['before' in x for x in summary_df.loc['hashtag']]]
    summary_between = summary_df.loc[:,
                     ['between' in x for x in summary_df.loc['hashtag']]]
    summary_after = summary_df.loc[:,
                     ['after' in x for x in summary_df.loc['hashtag']]]

    #%% Q7: aggregate all hashtags
    # cache agg data for future use
    hashtag_list = ['gohawks', 'gopatriots', 'patriots', 'sb49', 'nfl',
                    'superbowl']
    for i, period in enumerate(['before', 'between', 'after']):
        data = []
        for hashtag in hashtag_list:
            data.append(split_periods(hashtag)[i])
        data_agg = pd.concat(data, axis=0)
        data_agg.to_csv('data/aggregated_data_{}.csv'.format(period))
    #%% linear regression
    summary = []
    for i, period in enumerate(['before', 'between', 'after']):
        data_agg = pd.read_csv('data/aggregated_data_{}.csv'.format(period))
        data_agg['hour'] = pd.to_datetime(data_agg['hour'])
        X, y = extract_Xy(data_agg)
        summary.append(fit_OLS(X, y, 'aggregated_' + period))
    summary_df = pd.concat(summary, axis=1, ignore_index=True)

    #%% Q8: grid search
    param_grid = {'max_depth': [10, 20, 40, 60, 80, None],
                  'max_features': ['auto', 'sqrt'],
                  'min_samples_leaf': [1, 2, 4],
                  'min_samples_split': [2, 5, 10],
                  'n_estimators': [200, 400, 600, 800, 1000]}

    #%% random forest
    reg = RandomForestRegressor()
    grid = GridSearchCV(reg, param_grid=param_grid,
                        cv=KFold(5, shuffle=True, random_state=42),
                        scoring='neg_mean_squared_error', n_jobs=-1)
    summary_rf = []
    for i, period in enumerate(['before', 'between', 'after']):
        data_agg = pd.read_csv('data/aggregated_data_{}.csv'.format(period))
        data_agg['hour'] = pd.to_datetime(data_agg['hour'])
        X, y = extract_Xy(data_agg)
        grid.fit(X, y)
        summary_rf.append(grid.cv_results_)
    summary_before_rf, summary_between_rf, summary_after_rf = summary_rf

    #%% Q9: gradient boosting
    reg = GradientBoostingRegressor()
    grid = GridSearchCV(reg, param_grid=param_grid,
                        cv=KFold(5, shuffle=True, random_state=42),
                        scoring='neg_mean_squared_error', n_jobs=-1)
    summary_gb = []
    for i, period in enumerate(['before', 'between', 'after']):
        data_agg = pd.read_csv('data/aggregated_data_{}.csv'.format(period))
        data_agg['hour'] = pd.to_datetime(data_agg['hour'])
        X, y = extract_Xy(data_agg)
        grid.fit(X, y)
        summary_gb.append(grid.cv_results_)
    summary_before_gb, summary_between_gb, summary_after_gb = summary_gb















