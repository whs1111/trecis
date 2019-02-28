
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt


# In[2]:


import string
import gzip
import json
import re


# In[3]:


import numpy as np
import pandas as pd


# In[4]:


import scipy
from scipy import interpolate


# In[5]:


import sklearn.cluster
import sklearn.feature_extraction 
import sklearn.feature_extraction.text
import sklearn.metrics
import sklearn.preprocessing

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# In[6]:


import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

from nltk.tokenize import TweetTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer as VS


# In[7]:


import spacy

nlp = spacy.load('en')


# In[11]:


tweet_id_map = {}
with open("../data/rehydrated_tweets.json", "r") as in_file:
    for line in in_file:
        tweet = json.loads(line)
        tweet_id_map[tweet["id"]] = tweet


# In[12]:


lang_count_map = {}
for lang in [tweet["lang"] for tweet in tweet_id_map.values()]:
    lang_count_map[lang] = lang_count_map.get(lang, 0) + 1
lang_count_map


# In[13]:


tweet_category_map = {}
with open("../data/category_to_tweet_id_training.json", "r") as in_file:
    tweet_category_map = json.load(in_file)


# In[14]:


for category, tweet_ids in tweet_category_map.items():
    retrieved_count = sum([1 if int(tid) in tweet_id_map else 0 in tweet_id_map for tid in tweet_ids])
    print("Category:", category)
    print("\tTweet Count:", len(tweet_ids), "Retrieved Fraction:", retrieved_count/len(tweet_ids))
    
    lang_count_map = {}
    for lang in [tweet_id_map[int(tid)]["lang"] for tid in tweet_ids if int(tid) in tweet_id_map]:
        lang_count_map[lang] = lang_count_map.get(lang, 0) + 1
    print("\t", str(lang_count_map))


# In[15]:


# But first, read in stopwrods
enStop = stopwords.words('english')

# Skip stop words, retweet signs, @ symbols, and URL headers
stopList = enStop +    ["http", "https", "rt", "@", ":", "t.co", "co", "amp", "&amp;", "...", "\n", "\r"]
stopList.extend(string.punctuation)


# In[16]:


# def tokenizer_wrapper(text):
#     return [t.lemma_ for t in nlp(text)]

local_tokenizer = TweetTokenizer()
def tokenizer_wrapper(text):
    return local_tokenizer.tokenize(text)


# In[17]:


# Generate Additional Features
sentiment_analyzer = VS()

def count_twitter_objs(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE
    4) hashtags with HASHTAGHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned.
    
    Returns counts of urls, mentions, and hashtags.
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    return(parsed_text.count('URLHERE'),parsed_text.count('MENTIONHERE'),parsed_text.count('HASHTAGHERE'))

## Taken from Davidson et al.
def other_features(tweet):
    """This function takes a string and returns a list of features.
    These include Sentiment scores, Text and Readability scores,
    as well as Twitter specific features"""
    tweet_text = tweet["text"]
    
    ##SENTIMENT
    sentiment = sentiment_analyzer.polarity_scores(tweet_text)
    
    words = local_tokenizer.tokenize(tweet_text) #Get text only
    
    num_chars = sum(len(w) for w in words) #num chars in words
    num_chars_total = len(tweet_text)
    num_terms = len(tweet_text.split())
    num_words = len(words)
    num_unique_terms = len(set([x.lower() for x in words]))
    
    caps_count = sum([1 if x.isupper() else 0 for x in tweet_text])
    caps_ratio = caps_count / num_chars_total
    
    twitter_objs = count_twitter_objs(tweet_text) #Count #, @, and http://
    num_media = 0
    if "media" in tweet["entities"]:
        num_media = len(tweet["entities"]["media"])
    retweet = 0
    if "rt" in words or "retweeted_status" in tweet:
        retweet = 1
        
    has_place = 1 if "coordinates" in tweet else 0
        
    author = tweet["user"]
    is_verified = 1 if author["verified"] else 0
    log_followers = 0 if author["followers_count"] == 0 else np.log(author["followers_count"])
    log_friends = 0 if author["friends_count"] == 0 else np.log(author["friends_count"])
    
    features = [num_chars, num_chars_total, num_terms, num_words,
                num_unique_terms, sentiment['neg'], sentiment['pos'], 
                sentiment['neu'], sentiment['compound'],
                twitter_objs[2], twitter_objs[1],
                twitter_objs[0], retweet, num_media,
                is_verified, 
#                 log_followers, log_friends,
#                 has_place,
                caps_ratio,
               ]

    return features

other_features_names = ["num_chars", "num_chars_total",                         "num_terms", "num_words", "num_unique_words", "vader neg","vader pos",
                        "vader neu", "vader compound", \
                        "num_hashtags", "num_mentions", 
                        "num_urls", "is_retweet", "num_media",
                        "is_verified", 
#                         "log_followers", "log_friends",
#                         "has_place",
                        "caps_ratio",
                       ]


# In[18]:


vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
    tokenizer=tokenizer_wrapper,
    ngram_range=(1, 3),
    stop_words=stopList, #We do better when we keep stopwords
    use_idf=True,
    smooth_idf=False,
    norm=None, #Applies l2 norm smoothing
    decode_error='replace',
    max_features=10000,
    min_df=4,
    max_df=0.501
    )


# In[19]:


#We can use the TFIDF vectorizer to get a token matrix for the POS tags
pos_vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
    #vectorizer = sklearn.feature_extraction.text.CountVectorizer(
    tokenizer=None,
    lowercase=False,
    preprocessor=None,
    ngram_range=(1, 3),
    stop_words=None, #We do better when we keep stopwords
    use_idf=False,
    smooth_idf=False,
    norm=None, #Applies l2 norm smoothing
    decode_error='replace',
    max_features=5000,
    min_df=5,
    max_df=0.501,
    )


# In[33]:


category_to_label = {c:i+1 for i, c in enumerate(tweet_category_map.keys()) if c != "Irrelevant"}
category_to_label["Irrelevant"] = 0

tweet_id_to_category = {}
for category, tweet_ids in tweet_category_map.items():
    if ( len(tweet_ids) < 5 ):
        print("Skipping category:", category)
        continue
        
    for tweet_id in tweet_ids:
        tweet_id_to_category[int(tweet_id)] = category_to_label[category]


# In[34]:


tweet_pairs = [(tweet, tweet_id_to_category[tid]) 
               for tid, tweet in tweet_id_map.items() if tid in tweet_id_to_category]
tweet_texts = [tp[0]["text"] for tp in tweet_pairs]

y_data = np.array([tp[1] for tp in tweet_pairs])


# In[ ]:


#Construct tfidf matrix and get relevant scores
# tfidf = vectorizer.fit_transform(tweet_texts).toarray()

# vectorizer_ = joblib.load("../data/tfidf_vectorizer.pkl")
vectorizer_ = joblib.load("../data/2013to2016_tfidf_vectorizer.pkl")

# tmp_vect = vectorizer
tmp_vect = vectorizer_


# In[35]:


tfidf = tmp_vect.transform(tweet_texts).toarray()
vocab = {v:i for i, v in enumerate(tmp_vect.get_feature_names())}
idf_vals = tmp_vect.idf_
idf_dict = {i:idf_vals[i] for i in vocab.values()} #keys are indices; values are IDF scores


# In[36]:


#Construct POS TF matrix and get vocab dict
tweet_tags = [[t.tag_ for t in nlp(tp)] for tp in tweet_texts]
pos = pos_vectorizer.fit_transform(np.array([' '.join(x) for x in tweet_tags])).toarray()
pos_vocab = {v:i for i, v in enumerate(pos_vectorizer.get_feature_names())}


# In[37]:


other_ftr_data = np.array([other_features(tweet) for tweet, _ in tweet_pairs])


# In[38]:


X_data = np.concatenate([
    tfidf, 
    other_ftr_data, 
#     pos
], axis=1)

ftr_names_ = np.concatenate([
    np.array([x for x in vocab]), 
    other_features_names, 
#     np.array([x for x in pos_vocab.keys()])
])

print(X_data.shape, y_data.shape)


# In[26]:


print("Dictionary Word Count:", len(vocab))
print([x[0] for x in vocab.items()][-10:])


# In[27]:


r_state = 1337


# In[26]:


category_number_list = list(category_to_label.values())
d = len(category_number_list)

f1_accum = []
accuracy_accum = []

skf = StratifiedKFold(n_splits=5, random_state=r_state)
for train, test in skf.split(X_data, y_data):

    # Get actual labels
    y_test = y_data[test]

    # Compute Precision-Recall 
    y_infer_local = [category_number_list[np.random.randint(0, d)] for smaple in y_test]
    f1_accum.append(f1_score(y_test, y_infer_local, average="macro"))
    
    accuracy_accum.append(sklearn.metrics.accuracy_score(y_test, y_infer_local))

print("Accuracy:", np.mean(accuracy_accum))
print("F1:", np.mean(f1_accum))


# In[28]:


rf_params = {
    'n_estimators': 128, 
    "n_jobs": -1,
    'random_state': r_state,
    'class_weight': "balanced",
    'criterion': 'gini',
    'max_depth': 32,
    'max_features': 113,
    'min_samples_leaf': 2,
    'min_samples_split': 54,
}

nb_params = {
    'alpha': 0.01579145221181444,
    'binarize': 0.7316900686676242,
    'fit_prior': False
}


# In[32]:


f1_accum = []
accuracy_accum = []

skf = StratifiedKFold(n_splits=10, random_state=r_state)
for train, test in skf.split(X_data, y_data):

    X_train = X_data[train]
    y_train = y_data[train]
    
    X_test = X_data[test]
    y_test = y_data[test]

    # train
#     fitted_model = RandomForestClassifier(**rf_params)
    fitted_model = BernoulliNB(**nb_params)
    fitted_model.fit(X_train, y_train)

    # Compute Precision-Recall 
    y_infer_local = fitted_model.predict(X_test)
    f1_accum.append(f1_score(y_test, y_infer_local, average="macro"))
    
    accuracy_accum.append(fitted_model.score(X_test, y_test))

print("Accuracy:", np.mean(accuracy_accum))
print("F1:", np.mean(f1_accum))


# In[28]:


label_to_category = {j:i for i, j in category_to_label.items()}
for positive_category in category_number_list:
    local_y_data = [1 if y == positive_category else 0 for y in y_data]
    
    fitted_model = RandomForestClassifier(**rf_params)
    fitted_model.fit(X_data, local_y_data)
    
    weights = [(ftr_names_[idx], coef) 
               for idx, coef in enumerate(fitted_model.feature_importances_)]

    tops = sorted(weights, key=lambda x: x[1], reverse=True)[:10]
    
    print("Label:", label_to_category[positive_category])
    print("Score:", fitted_model.score(X_data, local_y_data))
    for token, weight in tops:
        print("\t", token, weight)


# In[29]:


def random_search(X_data, y_data, clf, param_dist, n_iter_search=20, r_state=1337):
    # run randomized search
    random_search = RandomizedSearchCV(clf, 
                                       param_distributions=param_dist,
                                       n_iter=n_iter_search,
                                       cv=10,
                                       scoring="f1_macro",
                                       random_state=r_state,
                                       verbose=2,
                                       n_jobs=-1,
                                      )
    
    random_search.fit(X_data, y_data)

    return (random_search.best_score_, random_search.best_params_)

def model_eval_rf(X_data, y_data, n_iter_search=100, r_state=1337):

    clf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=r_state)
    
    # specify parameters and distributions to sample from
    param_dist = {
        "max_depth": [2, 4, 8, 16, 32, 64, 128, None],
        "max_features": scipy.stats.randint(1, 512),
        "min_samples_split": scipy.stats.randint(2, 512),
        "min_samples_leaf": scipy.stats.randint(2, 512),
        "criterion": ["gini", "entropy"],
    }
    
    return random_search(X_data, y_data, clf, param_dist, n_iter_search=n_iter_search, r_state=r_state)

def model_eval_nb(X_data, y_data, n_iter_search=100, r_state=1337):

    clf = BernoulliNB()
    
    # specify parameters and distributions to sample from
    param_dist = {
        "alpha": scipy.stats.uniform(),
        "binarize": scipy.stats.uniform(),
        "fit_prior": [True, False],
    }
    
    return random_search(X_data, y_data, clf, param_dist, n_iter_search=n_iter_search, r_state=r_state)


# In[30]:


search_results = model_eval_nb(X_data, y_data, n_iter_search=128)
search_results


# In[31]:


search_results


# In[2]:


# Run baseline


# In[39]:


fitted_model = BernoulliNB(**nb_params)
fitted_model.fit(X_data, y_data)


# In[40]:


test_tweets = []
with gzip.open("../data/rehydrated_test_tweets.json.gz", "r") as in_file:
    for line_bytes in in_file:
        line = line_bytes.decode("utf8")
        tweet = json.loads(line)
        test_tweets.append(tweet)
        
X_test_tfidf = tmp_vect.transform([t["text"] for t in test_tweets]).toarray()
X_test_other = np.array([other_features(tweet) for tweet in test_tweets])

X_test_data = np.concatenate([
    X_test_tfidf, 
    X_test_other, 
], axis=1)

print(X_test_data.shape)


# In[41]:


y_test_labels = fitted_model.predict(X_test_data)


# In[42]:


y_test_labels[:100]


# In[44]:


labeled_test_data = list(zip([t["id"] for t in test_tweets], y_test_labels))


# In[47]:


id_to_cat_map = {y:x for x,y in category_to_label.items()}


# In[50]:


df = pd.DataFrame([{"tweet_id":tup[0], "label": id_to_cat_map[tup[1]]} for tup in labeled_test_data])
df.groupby("label").count()


# In[51]:


df.to_csv("trec2018_test_results_run_baseline.csv", index=None)

