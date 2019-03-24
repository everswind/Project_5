# %%
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
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import text
import nltk
from nltk import pos_tag
from sklearn.decomposition import NMF
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import itertools
from sklearn.cluster import KMeans
import re
import time


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
    tweet_per_hour = data.shape[0] / (span.days * 24 + span.seconds / 3600)
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

def extract_link(tweet):
    url = r'(https?://\S+)'
    tweet_string = json.dumps(tweet)
    urls = re.findall(url, tweet_string)
    url_count = len(urls)
    return url_count


def new_features(hashtag, question):

    filename = 'data/tweets_#{}.txt'.format(hashtag)
    citation_date = []
    author_followers = []
    citations = []
    url_count = []
    user_mentions = []
    ranking_score = []
    original_author_followers = []
    friends_count = []

    with open(filename, 'r') as f:
        for line in f:
            tweet = json.loads(line)
            citation_date.append(tweet['citation_date'])
            citations.append(tweet['metrics']['citations']['total'])
            author_followers.append(tweet['author']['followers'])
            original_author_followers.append(
                tweet['original_author']['followers'])
            user_mentions.append(
                len(tweet['tweet']['entities']['user_mentions']))
            friends_count.append(tweet['tweet']['user']['friends_count'])
            url_count.append(extract_link(tweet))
            ranking_score.append(tweet['metrics']['ranking_score'])

    data = pd.DataFrame({'citation_date': citation_date,
                         'citations': citations,
                         'author_followers': author_followers,
                         'original_author_followers': original_author_followers,
                         'user_mentions': user_mentions,
                         'friends_count': friends_count,
                         'url_count': url_count,
                         'ranking_score': ranking_score})

    data['utc_time'] = data['citation_date'].apply(datetime.datetime.utcfromtimestamp)
    data['utc_hour'] = data['utc_time'].apply(lambda t: t.replace(
        microsecond=0, second=0, minute=0))

    data_agg = data.groupby('utc_hour', as_index=False).agg(
        {'citations': ['count', 'sum', 'max'],
         'author_followers': ['sum', 'max'],
         'original_author_followers': ['sum', 'max'],
         'user_mentions': ['sum', 'max'],
         'friends_count': ['sum', 'max'],
         'url_count': ['sum', 'max'],
         'ranking_score': ['sum', 'max']})
    data_agg['utc_hour'] = data_agg['utc_hour'].apply(lambda x: x.hour)

    columns_ravel = lambda df: ['_'.join(x) for x in df.columns.ravel()]
    data_agg.columns = columns_ravel(data_agg)
    data_agg.rename(columns={'utc_hour_': 'hour_of_day',
                             'citations_count': 'tweets_sum',
                             'citations_sum': 'retweets_sum',
                             'citations_max': 'retweets_max'}, inplace=True)

    data_agg['target'] = data_agg['tweets_sum'].shift(-1)
    data_agg = data_agg.iloc[:-1]
    data_agg.to_csv('results/{}_#{}.csv'.format(question, hashtag))

    X, y = data_agg.drop(columns=['target']), data_agg['target']

    return X, y, data_agg


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
                            minute=t.minute - (t.minute % 5)))
    return before_df, between_df, after_df


def extract_period_data(filename):
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
                            minute=t.minute - (t.minute % 5)))
    return before_df, between_df, after_df


def gridsearch_periods(reg, param_grid):
    grid = GridSearchCV(reg, param_grid=param_grid,
                        cv=KFold(5, shuffle=True, random_state=42),
                        scoring='neg_mean_squared_error', n_jobs=-1)
    summary = []
    for i, period in enumerate(['before', 'between', 'after']):
        data_agg = pd.read_csv('data/aggregated_data_{}.csv'.format(period))
        data_agg['hour'] = pd.to_datetime(data_agg['hour'])
        X, y = extract_Xy(data_agg)
        grid.fit(X, y)
        results = grid.cv_results_
        results_df = pd.DataFrame(
            {'_'.join(['mean_test_score', period]): -results['mean_test_score'],
             '_'.join(['params', period]): results['params'],
             '_'.join(['rank', period]): results['rank_test_score']
             }).sort_values(by='_'.join(['rank', period])
                            ).reset_index(drop=True)
        summary.append(results_df)
    summary_df = pd.concat(summary, axis=1)
    return summary_df


def predict_6x(reg, title, period):
    mse_test = {}
    for s in [0, 1, 2]:
        file = 'data/ECE219_tweet_test/sample{}_period{}.txt'.format(
            s, period)
        X_test, y_test = extract_Xy(extract_period_data(file)[period - 1])
        pred = reg.predict(X_test)
        mse_test.update({'sample{}_period{}'.format(s, period):
                             metrics.mean_squared_error(y_test, pred)})
    mse_test_df = pd.DataFrame(mse_test, index=['mse_{}'.format(title)])
    return mse_test_df


def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN': 'n', 'JJ': 'a',
                  'VB': 'v', 'RB': 'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n'


def lemmatize(list_word):
    # lemmatize with Parts of Speech (POS) tags
    wnl = nltk.wordnet.WordNetLemmatizer()
    # Text input is string, returns array of lowercased strings(words).
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) for word, tag in
            pos_tag(list_word)]


def plot_confusion_matrix(cm, classes, ax,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')


def clf_report(clf, X, y, title):
    clf.fit(X, y)
    scores = clf.predict_proba(X)[:, 1]
    pred = clf.predict(X)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    fpr, tpr, t = metrics.roc_curve(y, scores)
    ax1.plot(fpr, tpr,
             label='{}, auc={:.4f}'.format(title, metrics.auc(fpr, tpr)))
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.legend()

    cm = metrics.confusion_matrix(y, pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    accuracy = (tp + tn) / np.sum(cm)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    plot_confusion_matrix(cm, ['Massachusetts', 'Washington'], ax=ax2,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues)
    plt.suptitle('{}, accuracy={:.4f},\nrecall={:.4f}, precision={:.4f}'.format(
        title, accuracy, recall, precision))
    plt.show()


def main():
    # %% Q1-2: first look
    results = []
    for hashtag in ['gohawks', 'gopatriots', 'patriots', 'sb49']:
        results.append(first_look(hashtag)[1])
    for hashtag in ['nfl', 'superbowl']:
        results.append(first_look(hashtag, plot=True)[1])
    results_df = pd.DataFrame(results)

    # %% Q3: linear regression
    hashtag_list = ['gohawks', 'gopatriots', 'patriots', 'sb49', 'nfl',
                    'superbowl']
    summary = []
    for hashtag in hashtag_list:
        data = first_look(hashtag)[0]
        X, y = extract_Xy(data)
        summary.append(fit_OLS(X, y, hashtag))
    summary_df = pd.concat(summary, axis=1, ignore_index=True)

    # %% Q4: design new features
    hashtag_list = ['gohawks', 'gopatriots', 'patriots', 'sb49', 'nfl',
                    'superbowl']
    summary = []
    est_params = []
    est_fitted_values = []
    Xs = []
    ys = []
    filename = 'Q4'
    for hashtag in hashtag_list:
        print('running #{}...'.format(hashtag))
        start_time = time.time()
        X, y, _ = new_features(hashtag, filename)
        sum, est = fit_OLS(X, y, hashtag)
        summary.append(sum)  # append dataframe
        est_params.append(est.params)
        est_fitted_values.append(est.fittedvalues)
        Xs.append(X)
        ys.append(y)
        print('time elapsed: {:.2f}s'.format(time.time() - start_time))

    summary_df = pd.concat(summary, axis=1, ignore_index=True)
    summary_df['mean'] = summary_df.sum(axis=1)
    summary_df['mean'] = summary_df['mean'].drop(
        index=['hashtag', 'mse_total', 'r_squared'])
    summary_df['mean'] = summary_df['mean'].astype(float).round(4) / 6
    est_params_df = pd.concat(est_params, axis=1, ignore_index=True)
    est_params_df.columns = hashtag_list

    hash_number = dict(gohawks=0, gopatriots=1, patriots=2, sb49=3, nfl=4,
                       superbowl=5)

    # %% Q5
    best_feature = ['url_count_sum', 'user_mentions_sum', 'retweets_max']

    for hashtag in hashtag_list:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        i = 0
        for feature in best_feature:
            slope = est_params[hash_number[hashtag]][feature]
            ax[i].scatter(Xs[hash_number[hashtag]][feature],
                          ys[hash_number[hashtag]], label='observed', s=15)
            ax[i].scatter(Xs[hash_number[hashtag]][feature],
                          est_fitted_values[hash_number[hashtag]],
                          label='fitted', s=15)
            ax[i].plot(Xs[hash_number[hashtag]][feature],
                       Xs[hash_number[hashtag]][feature] * slope,
                       'c', label='slope = {:.2f}'.format(slope))
            ax[i].set_title('#{}, feature: {}'.format(hashtag, feature),
                            fontsize=16)
            ax[i].set_xlabel('value of feature', fontsize=16)
            ax[i].set_ylabel('number of tweets for next hour', fontsize=16)
            ax[i].legend(loc='upper left', fontsize=12)
            ax[i].tick_params(labelsize=12)
            i += 1
        plt.tight_layout()
        plt.savefig('results/Q5_#{}.png'.format(hashtag), dpi=300)
        plt.show()

    # %% Q6: split time periods
    hashtag_list = ['gohawks', 'gopatriots', 'patriots', 'sb49', 'nfl',
                    'superbowl']
    summary = []
    for hashtag in hashtag_list:
        before_df, between_df, after_df = split_periods(hashtag)
        for i, data in enumerate([before_df, between_df, after_df]):
            X, y = extract_Xy(data)
            summary.append(
                fit_OLS(X, y,
                        hashtag + '_' + ['before', 'between', 'after'][i]))
    summary_df = pd.concat(summary, axis=1, ignore_index=True)
    summary_before = summary_df.loc[:,
                     ['before' in x for x in summary_df.loc['hashtag']]]
    summary_between = summary_df.loc[:,
                      ['between' in x for x in summary_df.loc['hashtag']]]
    summary_after = summary_df.loc[:,
                    ['after' in x for x in summary_df.loc['hashtag']]]

    # %% Q7: aggregate all hashtags
    # cache agg data for future use
    hashtag_list = ['gohawks', 'gopatriots', 'patriots', 'sb49', 'nfl',
                    'superbowl']
    for i, period in enumerate(['before', 'between', 'after']):
        data = []
        for hashtag in hashtag_list:
            data.append(split_periods(hashtag)[i])
        data_agg = pd.concat(data, axis=0)
        data_agg.to_csv('data/aggregated_data_{}.csv'.format(period))
    # %% linear regression
    summary = []
    for i, period in enumerate(['before', 'between', 'after']):
        data_agg = pd.read_csv('data/aggregated_data_{}.csv'.format(period))
        data_agg['hour'] = pd.to_datetime(data_agg['hour'])
        X, y = extract_Xy(data_agg)
        summary.append(fit_OLS(X, y, 'aggregated_' + period))
    summary_df = pd.concat(summary, axis=1, ignore_index=True)

    # %% Q8: random forest, grid search
    param_grid = {'max_depth': [10, 20, 40, 60, 80, None],
                  'max_features': ['auto', 'sqrt'],
                  'min_samples_leaf': [1, 2, 4],
                  'min_samples_split': [2, 5, 10],
                  'n_estimators': [200, 400, 600, 800, 1000]}
    reg = RandomForestRegressor()
    summary_df_rf = gridsearch_periods(reg, param_grid)

    # %% Q9: compare RF and OLS on entire agg dataset
    data_agg_all = []
    for i, period in enumerate(['before', 'between', 'after']):
        data_agg = pd.read_csv('data/aggregated_data_{}.csv'.format(period))
        data_agg['hour'] = pd.to_datetime(data_agg['hour'])
        data_agg_all.append(data_agg)
    data_agg_all = pd.concat(data_agg_all, axis=0, ignore_index=True)
    X, y = extract_Xy(data_agg_all)
    rf = RandomForestRegressor(max_depth=80, max_features='sqrt',
                               min_samples_leaf=4, min_samples_split=5,
                               n_estimators=200)
    rf.fit(X, y)
    y_pred = rf.predict(X)
    mse_rf = metrics.mean_squared_error(y, y_pred)
    mse_OLS = fit_OLS(X, y, 'total').loc['mse_total', 0]
    print('On entire aggregated data, random forest mse: {:.4f}'.format(mse_rf),
          ', OLS mse: {:.4f}'.format(mse_OLS))

    # %% Q10: gradient boosting, grid search
    param_grid = {'max_depth': [10, 20, 40, 60, 80, None],
                  'max_features': ['auto', 'sqrt'],
                  'min_samples_leaf': [1, 2, 4],
                  'min_samples_split': [2, 5, 10],
                  'n_estimators': [200, 400, 600, 800, 1000]}
    reg = GradientBoostingRegressor()
    summary_df_gb = gridsearch_periods(reg, param_grid)

    # %% Q11: MLPRegressor
    X, y = extract_Xy(data_agg_all)
    size_list = [(50,), (100,), (200,), (20, 10), (50, 10)]
    mse_nn = {}
    for size in size_list:
        reg = MLPRegressor(hidden_layer_sizes=size)
        reg.fit(X, y)
        mse_nn.update(
            {str(size): [metrics.mean_squared_error(y, reg.predict(X))]})
    mse_nn_df = pd.DataFrame(mse_nn)

    # %% Q12: standard scalar
    scaler = StandardScaler()
    X_scale = scaler.fit_transform(X)
    reg = MLPRegressor(hidden_layer_sizes=(200,))
    reg.fit(X_scale, y)
    print(metrics.mean_squared_error(y, reg.predict(X_scale)))

    # %% Q13: grid search for periods
    param_grid = {'hidden_layer_sizes': size_list}
    reg = MLPRegressor()
    summary_df_nn = gridsearch_periods(reg, param_grid)

    # %% Q14: train a rf using all agg data first
    data_agg_all = []
    for i, period in enumerate(['before', 'between', 'after']):
        data_agg = pd.read_csv('data/aggregated_data_{}.csv'.format(period))
        data_agg['hour'] = pd.to_datetime(data_agg['hour'])
        data_agg_all.append(data_agg)
    data_before, data_between, data_after = data_agg_all
    data_agg_all = pd.concat(data_agg_all, axis=0, ignore_index=True)
    X_all, y_all = extract_Xy(data_agg_all)
    X_before, y_before = extract_Xy(data_before)
    X_between, y_between = extract_Xy(data_between)
    X_after, y_after = extract_Xy(data_after)

    rf = RandomForestRegressor(max_depth=80, max_features='sqrt',
                               min_samples_leaf=4, min_samples_split=5,
                               n_estimators=200)
    # load test data and predict
    rf.fit(X_all, y_all)
    mse_test_all_1 = predict_6x(rf, 'rf_all', 1)
    mse_test_all_2 = predict_6x(rf, 'rf_all', 2)
    mse_test_all_3 = predict_6x(rf, 'rf_all', 3)

    rf.fit(X_before, y_before)
    mse_test_before_1 = predict_6x(rf, 'rf_before', 1)
    rf.fit(X_between, y_between)
    mse_test_between_2 = predict_6x(rf, 'rf_between', 2)
    rf.fit(X_after, y_after)
    mse_test_after_3 = predict_6x(rf, 'rf_after', 3)

    # %% Q15 fan base prediction, label base first
    filename = 'data/tweets_#superbowl.txt'
    substring_WA = ['WA', 'Washington', 'Seattle']
    substring_MA = ['MA', 'Massachusetts', 'Boston']
    locations = []
    base = []
    text = []
    with open(filename, 'r') as f:
        for line in f:
            line = json.loads(line)
            location = line['tweet']['user']['location']
            if any([(s in location) and ('DC' not in location)
                    for s in substring_WA]):
                base.append('Washington')
            elif any([s in location for s in substring_MA]):
                base.append('Massachusetts')
            else:
                base.append('other')
            locations.append(location)
            text.append(line['tweet']['text'])
    data = pd.DataFrame({'location': locations, 'base': base,
                         'text': text})
    data_location = data[data['base'] != 'other'].reset_index(drop=True)

    # %% base classification
    # create custom analyzer remove_num, incoporating the lemmatizer
    analyze = CountVectorizer().build_analyzer()  # default analyzer
    remove_num = lambda doc: [word for word in lemmatize(analyze(doc))
                              if not word.isdigit()]
    # create vectorizer with the above analyzer, vectorize and tfidf_transform
    vectorizer = CountVectorizer(min_df=3, analyzer=remove_num,
                                 stop_words='english')
    tfidf_transformer = TfidfTransformer(smooth_idf=False)
    # fit and transform on train data
    y = data_location['base']
    y[y == 'Massachusetts'] = 0
    y[y == 'Washington'] = 1
    y = np.array(y).astype(int)

    X = vectorizer.fit_transform(data_location['text'])
    X_tfidf = tfidf_transformer.fit_transform(X)

    # %% reduce dimension then classify
    nmf = NMF(n_components=50, init='random', random_state=0)
    X_tfidf_nmf = nmf.fit_transform(X_tfidf)

    # %% run clf, report ROC, confusion matrix, accuracy, recall, precision
    # %% Naive Bayes
    mnb = MultinomialNB(alpha=0.01)
    clf_report(mnb, X_tfidf_nmf, y, 'Multinomial Naive Bayes')
    # %% Logistic
    lr = LogisticRegression(C=1e6)
    clf_report(lr, X_tfidf_nmf, y, 'Logistic Regression')
    # %% random forest
    rfc = RandomForestClassifier(500, max_depth=10, max_features='auto')
    clf_report(rfc, X_tfidf_nmf, y, 'Random Forest')
    # %% 5 fold CV
    cv_clf = []
    for clf in [mnb, lr, rfc]:
        cv_clf.append(np.mean(
            cross_validate(clf, X_tfidf_nmf, y,
                           cv=5, scoring='accuracy', n_jobs=-1)['test_score']))
    print('5 fold CV, mean test accuracy, mnb={:.4f}, lr={:.4f}'
          ', rfc={:.4f}'.format(*cv_clf))

    # %% Q16 predict support team for people from other areas
    data_other = data[data['base'] == 'other'].reset_index(drop=True)
    X_other = vectorizer.transform(data_other['text'])
    X_other_tfidf = tfidf_transformer.transform(X_other)
    X_other_nmf = nmf.transform(X_other_tfidf)
    lr.fit(X_tfidf_nmf, y)
    pred_other = lr.predict(X_other_nmf)

    #%% visualize
    for i,team in enumerate(['Patriots', 'Seahawks']):
        X_team = X_other_nmf[pred_other == i, :]
        plt.scatter(X_team[:, 0], X_team[:, 1], marker='.', alpha=0.5,
                    label='{}, {}'.format(team, len(X_team)))
    plt.legend()
    plt.show()

    #%% cluster MA, WA
    cluster = KMeans(n_clusters=6, n_init=30, max_iter=1000,
                     random_state=0, n_jobs=-1)
    pred = cluster.fit_predict(X_tfidf_nmf)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(X_tfidf_nmf[:, 0], X_tfidf_nmf[:, 1], marker='.', alpha=0.5,
                    c=pred, label='clustered, log then scale')
    axes[0].legend(loc="upper right")
    axes[1].scatter(X_tfidf_nmf[:, 0], X_tfidf_nmf[:, 1], marker='.', alpha=0.5,
                    c=y, label='labeled')
    axes[1].legend(loc="upper right")
    plt.show()
