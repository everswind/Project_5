#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pytz
import json
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


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
    data['hour'] = data['time'].apply(lambda t: t.replace(microsecond=0, second=0,
                                                          minute=0))
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


def extract_Xy(hashtag, feature_func):
    data = feature_func(hashtag)[0]
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


def fit_OLS(hashtag_list, feature_func):
    summary = []
    for hashtag in hashtag_list:
        X, y = extract_Xy(hashtag, feature_func)
        X_c = sm.add_constant(X)
        reg = sm.OLS(y, X_c)
        est = reg.fit()
        summary.append(dict(hashtag=hashtag,
                            mse_total=est.mse_total,
                            r_squared=est.rsquared,
                            tvalues=dict(est.tvalues),
                            pvalues=dict(est.pvalues)))
    summary_df = pd.DataFrame(summary)
    # expand tvalues/pvalues into columns
    for column in ['tvalues', 'pvalues']:
        df_expand = pd.DataFrame(summary_df[column].values.tolist())
        df_expand.rename(columns=lambda x: '_'.join([column, x]), inplace=True)
        summary_df = pd.concat([summary_df.drop(columns=column),
                                df_expand], axis=1)
    summary_df = summary_df.T
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
    data['time'] = data['date'].apply(datetime.datetime.utcfromtimestamp)
    data['hour'] = data['time'].apply(lambda t: t.replace(microsecond=0, second=0,
                                                          minute=0))
    span = data['hour'].max() - data['hour'].min()
    tweet_per_hour = data.shape[0] / (span.days*24 + span.seconds/3600)
    average_followers = data['followers'].mean()
    average_retweets = data['citation'].mean()

    result = {'hashtag': hashtag,
              'average tweets per hour': tweet_per_hour,
              'average followers per tweet': average_followers,
              'average retweets per tweet': average_retweets}

    return data, result


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
    summary_df = fit_OLS(hashtag_list, first_look)

    #%% Q4: design new features







