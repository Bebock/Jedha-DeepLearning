#!/usr/bin/env python
# coding: utf-8

# # **Natural Language Processing with Disaster Tweets**
# 
# https://www.kaggle.com/c/nlp-getting-started

# **Context**
# Twitter has become an important communication channel in times of emergency.
# The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).
# 
# In this competition, you’re challenged to build a machine learning model that predicts which Tweets are about real disasters and which one’s aren’t. You’ll have access to a dataset of 10,000 tweets that were hand classified.

# # 1. Imports & Loading

# In[1]:


# Usual tools

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pprint import pprint
import pickle
import seaborn as sns
sns.set(style='white', context='notebook', palette='deep') 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.style as style
style.use('fivethirtyeight')
import math
import missingno
from bioinfokit.analys import stat
import warnings 
warnings.filterwarnings('ignore')
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image

# Text processing 

import collections
from collections import Counter
from unidecode import unidecode
import re
import string
from spellchecker import SpellChecker
import contractions

# NLP & LDA

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
nlp = spacy.load('en_core_web_sm')
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.utils import simple_preprocess
from gensim import corpora
import gensim
import gensim.corpora as corpora
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
import pyLDAvis
import pyLDAvis.gensim
from wordcloud import WordCloud
from textblob import TextBlob

# Topic analysis

from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# sklearn : Preprocessing 

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

# Model Selection & evaluation

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, BayesianRidge
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, precision_score, accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, recall_score, classification_report
from shapash.explainer.smart_explainer import SmartExplainer
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier, ElasticNet
from xgboost import XGBClassifier

# Tensorflow

import tensorflow as tf
import tensorflow_hub as hub # USE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_addons as tfa


# In[2]:


# Load datasets

train_df = pd.read_csv("train.csv", encoding = "ISO8859-1")
print(f'Nb of rows in TRAIN : {len(train_df)}')


# In[3]:


# Dealing with encoding

train_df['text'] = train_df['text'].apply(unidecode)
train_df['keyword'] = [unidecode(x) if x is pd.notnull else x for x in train_df['keyword'] ]


# In[4]:


print(f'Nb of duplicated tweets deleted from the dataset : {sum(train_df["text"].duplicated())}')
train_df = train_df.drop_duplicates(subset = ['text']).reset_index()


# # 2. Exploratory Data Analysis

# ## 2.1. Target variable

# In[5]:


print(f'The dataset counts {train_df.target.value_counts()[0]} ({round(train_df.target.value_counts()[0]/len(train_df)*100, 2)}%) Tweets not signaling a disaster against {train_df.target.value_counts()[1]} ({round(train_df.target.value_counts()[1]/len(train_df)*100, 2)}%) Tweets signaling one. ')
sns.countplot(train_df.target);


# Data are quite balanced regarding the target to predict. 

# ## 2.2. Location variable

# In[6]:


train_df.location.value_counts()


# In[7]:


print(f"We count {sum(train_df['location'].isna())} Tweets with missing location - around {round(sum(train_df['location'].isna())/len(train_df)*100,2)}% of the dataset")


# The proportion of missing location is quite high. To this percent, we also have to add the tweets which are only located thanks to the country or the continent, and the tweets voluntariliry hiding its location (Asgard, Mars, In the shadow, Somewhere only we know ...). 
# Thus, we will not use this variable. It would have had a gread value to predict the fact that a tweet signals or not a disaster (crossing location and time to see the co occurrence of several tweets in the same area in a short time). 

# ## 2.3. Keywords Variable 

# In[8]:


train_df.keyword.value_counts()


# In[9]:


print(f'Nb of missing data in the Keywords variable : {sum(train_df["keyword"].isna())} ({round(sum(train_df["keyword"].isna()) / len(train_df) * 100, 0)}%)')


# ## 2.4. Extraction of additional descriptive features

# In[10]:


# Writing features
train_df['Nb_char'] = train_df['text'].str.len()
train_df['Nb_word'] = train_df['text'].str.split().apply(len)
train_df['Avg_word_len'] = train_df['text'].str.split().apply(lambda x : [len(i) for i in x]).map(lambda x : np.mean(x))
train_df = train_df.assign(Nb_Exclam = [x.count("!") for x in train_df.text])
train_df = train_df.assign(Nb_Upper = [len([elem for elem in x if elem.isupper()]) for x in train_df.text])

# Presence and number of #
train_df = train_df.assign(Hashtag = ["#" in x for x in train_df.text])
train_df = train_df.assign(Nb_Hashtag = [x.count("#") for x in train_df.text])

# Presence and number of @
train_df = train_df.assign(At = ["@" in x for x in train_df.text])
train_df = train_df.assign(Nb_At = [x.count("@") for x in train_df.text])

# Presence and number of urls
train_df = train_df.assign(Link = ["http" in x or "www" in x for x in train_df.text])
train_df = train_df.assign(Nb_link = [x.count("http") + x.count("www") for x in train_df.text])


# ### What about the quantitative features ?

# In[11]:


quant_var = ['Nb_char', 'Nb_word', 'Avg_word_len', 'Nb_Exclam', 'Nb_Upper', 'Nb_Hashtag', 'Nb_At', 'Nb_link']
for var in quant_var : 
    res = stat()
    formula = 'target ~ C(' + var + ')'
    res.anova_stat(df = train_df, res_var = 'target', anova_model = formula)
    print('Effect on {} - p value  = {}'.format(var, round(res.anova_summary['PR(>F)'][0],3)))


# In[12]:


quant_var = ['target', 'Nb_char', 'Nb_word', 'Avg_word_len', 'Nb_Exclam', 'Nb_Upper', 'Nb_Hashtag', 'Nb_At', 'Nb_link']
sns.pairplot(train_df[quant_var], hue = 'target', corner = True);
plt.suptitle('Bivariate analysis of quantitative features', size = 16);


# In[13]:


quant_var = ['Nb_char', 'Nb_word', 'Avg_word_len', 'Nb_Exclam', 'Nb_Upper', 'Nb_Hashtag', 'Nb_At', 'Nb_link']
    
plt.figure(figsize = (12, 12))
plt.subplots_adjust(hspace = 0.5)
plt.suptitle("Quantitatives features according to the target variable", fontsize = 18)

for n, ticker in enumerate(quant_var):
    
    ax = plt.subplot(4, 4, n + 1) # add a new subplot iteratively

    plt.bar(train_df.target, train_df[ticker])

    ax.set_title(ticker.upper())
    ax.set_xlabel("")


# In[14]:


quant_var = ['Nb_char', 'Nb_word', 'Avg_word_len', 'Nb_Exclam', 'Nb_Upper', 'Nb_Hashtag', 'Nb_At', 'Nb_link']
    
plt.figure(figsize = (12, 12))
plt.subplots_adjust(hspace = 0.5)
plt.suptitle("Quantitatives features according to the target variable", fontsize = 18)

for n, ticker in enumerate(quant_var):
    
    ax = plt.subplot(4, 4, n + 1) # add a new subplot iteratively

    sns.histplot(train_df, x = ticker, stat = 'percent', hue = "target");

    ax.set_title(ticker.upper())
    ax.set_xlabel("")


# ### What about the presence / absence features ?

# In[15]:


qual_var = ['Hashtag', 'At', 'Link']
for var in qual_var : 
    res = stat()
    formula = 'target ~ C(' + var + ')'
    res.anova_stat(df = train_df, res_var = 'target', anova_model = formula)
    print('Effect on {} - p value  = {}'.format(var, round(res.anova_summary['PR(>F)'][0],3)))


# In[16]:


pd.crosstab(index = train_df['target'], columns = train_df['Hashtag'], normalize = 'index').plot.bar()
pd.crosstab(index = train_df['target'], columns = train_df['At'], normalize = 'index').plot.bar()
pd.crosstab(index = train_df['target'], columns = train_df['Link'], normalize = 'index').plot.bar()


# # 3. Preprocessing

# In[17]:


# Lower casing

train_df['cleaned'] = train_df['text'].apply(lambda x: x.lower())

# Replace &amp; with &

train_df['cleaned'] = train_df['cleaned'].str.replace('&amp;','')

# Expanding English Contractions

train_df['cleaned'] = train_df['cleaned'].apply(lambda x : contractions.fix(x))

# Remove http / https URLs

train_df['cleaned'] = train_df['cleaned'].str.replace(r'http\S+', '', regex = True)
train_df['cleaned'] = train_df['cleaned'].apply(lambda x : re.sub(r'www\.[^\s]+', '', x)) 

# Remove emails

train_df['cleaned'] = train_df['cleaned'].apply(lambda x : re.sub(r'\S*@\S*\s?', '', x))

# Remove digits and words containing digits 

train_df['cleaned'] = train_df['cleaned'].apply(lambda x: re.sub('\w*\d\w*','', x))

# Remove usernames @

train_df['cleaned'] = train_df['cleaned'].apply(lambda x : re.sub(r'@[^ ]+', '', x))

# Character normalization (repeated letters such as yesssss)

train_df['cleaned'] = train_df['cleaned'].apply(lambda x : re.sub(r'([A-Za-z])\1{2,}', r'\1', x))

# Remove special characters

def tweet_cleaner(tweet):
    tweet = re.sub(r"%20", " ", tweet)
    tweet = re.sub(r"%", " ", tweet)
    tweet = re.sub(r"'", " ", tweet)
    tweet = re.sub(r"\x89û_", " ", tweet)
    tweet = re.sub(r"\x89ûò", " ", tweet)
    tweet = re.sub(r"re\x89û_", " ", tweet)
    tweet = re.sub(r"\x89û", " ", tweet)
    tweet = re.sub(r"\x89Û", " ", tweet)
    tweet = re.sub(r"re\x89Û", "re ", tweet)
    tweet = re.sub(r"re\x89û", "re ", tweet)
    tweet = re.sub(r"\x89ûª", "'", tweet)
    tweet = re.sub(r"\x89û", " ", tweet)
    tweet = re.sub(r"\x89ûò", " ", tweet)
    tweet = re.sub(r"\x89Û_", "", tweet)
    tweet = re.sub(r"\x89ÛÒ", "", tweet)
    tweet = re.sub(r"\x89ÛÓ", "", tweet)
    tweet = re.sub(r"\x89ÛÏ", "", tweet)
    tweet = re.sub(r"\x89Û÷", "", tweet)
    tweet = re.sub(r"\x89Ûª", "", tweet)
    tweet = re.sub(r"\x89Û\x9d", "", tweet)
    tweet = re.sub(r"å_", "", tweet)
    tweet = re.sub(r"\x89Û¢", "", tweet)
    tweet = re.sub(r"\x89Û¢åÊ", "", tweet)
    tweet = re.sub(r"åÊ", "", tweet)
    tweet = re.sub(r"åÈ", "", tweet)  
    tweet = re.sub(r"Ì©", "e", tweet)
    tweet = re.sub(r"å¨", "", tweet)
    tweet = re.sub(r"åÇ", "", tweet)
    tweet = re.sub(r"åÀ", "", tweet)
    tweet = re.sub(r"â", "", tweet)
    tweet = re.sub(r"ã", "", tweet)
    return tweet

train_df['cleaned'] = train_df['cleaned'].apply(lambda x: tweet_cleaner(x))

# Remove punctuation 

train_df['cleaned'] = train_df['cleaned'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))

# Remove double spaces

train_df['cleaned'] = train_df['cleaned'].apply(lambda x : re.sub(r'  ', ' ', x))

# Remove \n

train_df['cleaned'] = train_df['cleaned'].apply(lambda x : re.sub(r'\n', '', x))

# Correct spelling (https://pypi.org/project/pyspellchecker/)

spell = SpellChecker()
for i in range(len(train_df)) : 
    text = train_df['cleaned'][i]
    corrected = []
    misspelled = spell.unknown(text.split())
    if len(misspelled) > 0 :
        for word in text.split():
            if word in misspelled :
                if spell.correction(word) is None :
                    corrected.append(word)
                else :
                    corrected.append(spell.correction(word))
            else:
                corrected.append(word)
        train_df['cleaned'][i] = " ".join(corrected)


# In[18]:


# Tokenization & Lemmatization (removing stopwords)

tokens = []
lemma = []
pos = []

for doc in nlp.pipe(train_df["cleaned"].astype('unicode').values, batch_size=50):
    tokens.append([n.text for n in doc if ((n.lemma_ not in STOP_WORDS) and (len(n.text) >= 2))])
    lemma.append([n.lemma_ for n in doc if ((n.lemma_ not in STOP_WORDS) and (len(n.lemma_) >= 2))])

train_df['desc_tokens'] = tokens
train_df['desc_lemma'] = lemma
train_df['desc_lemma_text'] = train_df['desc_lemma'].map(lambda x: " ".join(s for s in x if len(s) >= 2))


# # 4. Content description

# ## 4.1. Top n-grams

# In[19]:


def get_top_ngram(corpus, n = None):
    vec = CountVectorizer(ngram_range = (n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis = 0) 
    words_freq = [(word, sum_words[0, idx]) 
                  for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse =  True)
    return words_freq[:10]


# In[20]:


top_n_bigrams_0 = get_top_ngram(train_df[(train_df.target == 0)]['desc_lemma_text'], 2)[:10]
x0, y0 = map(list, zip(*top_n_bigrams_0))
top_n_bigrams_1 = get_top_ngram(train_df[(train_df.target == 1)]['desc_lemma_text'], 2)[:10]
x1, y1 = map(list, zip(*top_n_bigrams_1))

fig, axs = plt.subplots(ncols = 2, figsize = (15, 7))
sns.barplot(x = y0, y = x0, ax = axs[0]).set(title = 'Top 2-grams for Not disaster Tweets')
sns.barplot(x = y1, y = x1, ax = axs[1]).set(title = 'Top 2-grams for Disaster Tweets')


# ## 4.2. Wordclouds

# In[21]:


text_merge_clean_0 = ''
for j in range(0, len(train_df[train_df['target'] == 0])):
    text_merge_clean_0 += str(train_df[train_df['target'] == 0]['desc_lemma_text'].iloc[j])
text_merge_clean_1 = ''
for j in range(0, len(train_df[train_df['target'] == 1])):
    text_merge_clean_1 += str(train_df[train_df['target'] == 1]['desc_lemma_text'].iloc[j])

mask = np.array(Image.open("twitter.png"))
fig = plt.figure(figsize = (20, 10))
ax = fig.add_subplot(1,2,1)
plt.subplot(1,2,1).set_title('Wordcloud for Not disaster Tweets')
ax.imshow(WordCloud(mask = mask, contour_width = 1, contour_color = "grey", min_font_size = 3, max_words = 50, width = 1600, height = 800, stopwords = STOP_WORDS).generate(text_merge_clean_0))
plt.grid(None)
ax = fig.add_subplot(1,2,2)
plt.subplot(1,2,2).set_title('Wordcloud for Disaster Tweets')
ax.imshow(WordCloud(mask = mask, contour_width = 1, contour_color = "grey", min_font_size = 3, max_words = 50, width = 1600, height = 800, stopwords = STOP_WORDS).generate(text_merge_clean_1))
plt.grid(None)


# ## 4.3. Sentiment analysis

# In[22]:


sid = SentimentIntensityAnalyzer()

def get_vader_score(sent):
    # Polarity score returns dictionary
    ss = sid.polarity_scores(sent)
    #return ss
    return np.argmax(list(ss.values())[:-1])

train_df['polarity'] = train_df['desc_lemma_text'].map(lambda x: get_vader_score(x))

train_df['polarity'] = train_df['polarity'].replace({0 : 'negative', 1 : 'neutral', 2 : 'positive'})

sns.countplot(x = 'polarity', hue = 'target', data = train_df);


# In[23]:


set(train_df[(train_df['target'] == 1) & (train_df['polarity'] == 'positive')]['text'])


# Reading as these tweets, we can question some of the coding. For ex
# 
# my favorite lady came to our volunteer meeting hopefully joining her youth collision and i am excite 
# ok peace I hope I fall off a cliff along with my dignity
# :) well I think that sounds like a fine plan where little derailment is possible so I applaud you :)

# ## 4.4. Topic modelling
# 
# The aim here is to classify the Tweets content according to its topic. The technique the most affordable is an unsupervised technique because it would be too time consuming to manually label all topics represented in each tweet. Topic Modeling is an unsupervised learning method as it automatically groups words without any predefined labels / classifications. 
# 
# With LDA, each document (Tweet) is considered as a collection of topics in a certain proportion and each topic is a collection of keywords in a certain proportion. 
# 
# Doing so, a topic is no more than a collection of frequent keywords (representatives). It is the user's job to interpret the topic meaning thanks to all the keywords. 

# Topic coherence measures the average similarity between top words having the highest weights in a topic (= distance between the top words). We will use this score to choose the best model parameters. 

# In[24]:


# LDA model training - choosing optimal nb of topics

id2word = corpora.Dictionary(train_df['desc_lemma']) 
texts = train_df['desc_lemma'] 
corpus = [id2word.doc2bow(text) for text in texts]


# In[25]:


coherence_values = []
model_list = []
for num_topics in range(1,20,4):
    model = gensim.models.LdaMulticore(random_state = 60, 
                                       corpus = corpus,
                                       id2word = id2word,
                                       num_topics = num_topics)
    model_list.append(model)
    coherence_values.append(CoherenceModel(model = model, texts = train_df['desc_lemma'], dictionary = id2word, coherence = 'c_v').get_coherence()) 


# In[26]:


plt.plot(range(1,20,4), coherence_values)
plt.xlabel("Number of topics")
plt.ylabel("Coherence score")
plt.show() 


# In[27]:


# LDA model training - choosing optimal hyperparameters

a_values = ['symmetric', 'asymmetric']
e_values = ['symmetric', 'auto']
coherence_values = []
model_list = []
for alpha in a_values:
    for eta in e_values :
        model = gensim.models.LdaMulticore(random_state = 60, 
                                            corpus = corpus,
                                            id2word = id2word,
                                            num_topics = 7, 
                                            alpha = alpha, 
                                            eta = eta)
        model_list.append(model)
        coherence_values.append(CoherenceModel(model = model, texts = train_df['desc_lemma'], dictionary = id2word, coherence = 'c_v').get_coherence()) 


# In[28]:


i = 0
for alpha in a_values:
    for eta in a_values :
        print("Model with Alpha =", alpha, "and Eta =", eta, "has Coherence value of", round(coherence_values[i], 3))
        i += 1


# In[29]:


# Final LDA model training

lda_model = gensim.models.LdaMulticore(random_state = 60, 
                                       corpus = corpus,
                                       id2word = id2word,
                                       num_topics = 7, 
                                       alpha = 'asymmetric', 
                                       eta = 'symmetric')

pprint(lda_model.print_topics())

doc_lda = lda_model[corpus]


# Each topic is represented by an equation in which each keyword is weighted (importance) to indicate how much it contributes to the topic. How to interpret this kind of output ?
# 
# Topic 0 is a represented as 0.008*"like" + 0.006*"fire" + 0.004*"day" + 0.004*"people" + 0.004*"news" + 0.003*"crash" + 0.003*"video" + 0.003*"love" + 0.003*"kill" + '0.003*"come".
# 
# So this topic is represented by the top 2 keywords are like, fire. 

# In[30]:


# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus)) 

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model = lda_model, texts = train_df['desc_lemma'], dictionary = id2word, coherence = 'c_v')
print('\nCoherence Score: ', coherence_model_lda.get_coherence())


# In[31]:


# Analyzing LDA model results

pyLDAvis.enable_notebook()

LDAvis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
pyLDAvis.display(LDAvis)


# We observe 2 main topics (T1 and T2), close to each others in the left of PC1 with a third topic (T3). 
# - T1 : like, fire
# - T2 : like, fire
# - T3 : family, new, legionnaire
# 
# At the opposite side of PC1 , some little topic 
# - T6 : like, wreck
# - T7 : reddit, quarantine
# 
# The topics T4 and T5 are contrasted along PC2.
# - T4 : new, crash
# - T5 : fire, water

# In[32]:


cloud = WordCloud(stopwords = STOP_WORDS, width = 4000, height = 4000)
topics = lda_model.show_topics(formatted = False)

fig, axes = plt.subplots(2,4, figsize = (10,7))
for i, ax in enumerate(axes.flatten()):
    if i < 7 :
        fig.add_subplot(ax)
        words = dict(topics[i][1])
        cloud.generate_from_frequencies(words)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i + 1))
        plt.gca().axis('off')

fig.delaxes(axes[1][3])
plt.show()


# In[33]:


# Get topic weights
topic_weights = []
for i, row_list in enumerate(lda_model[corpus]):
    topic_weights.append([w for i, w in row_list])

# Array of topic weights    
arr = pd.DataFrame(topic_weights).fillna(0).values

# Keep the well separated points (optional)
arr = arr[np.amax(arr, axis=1) > 0.35]

# Dominant topic number in each doc
topic_num = np.argmax(arr, axis=1)

# tSNE Dimension Reduction
tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
tsne_lda = tsne_model.fit_transform(arr)


# In[34]:


# Plot the Topic Clusters using Bokeh
output_notebook()
n_topics = 6
mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), 
              plot_width=900, plot_height=700)
plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])
show(plot)


# In[35]:


train_df['topic'] = ""
for i in range(len(train_df)) :
    train_df['topic'][i] = np.argmax(topic_weights[i])


# In[36]:


pd.crosstab(index = train_df['target'], columns = train_df['topic'], normalize = 'index').plot.bar()


# Main topic identification does not seem to distinguish the disaster tweets from others.  

# # 5. Models comparison : Could we predict the type of a tweet - disaster related or not ? 

# ## 5.1. Classifiers - Basic features & Topics
# 
# First of all, we tried traditional Machine Learning algorithms on text features. 
# We tried to predict the nature of the tweets using the descriptive features and topics extracted before.  

# In[37]:


X = train_df.loc[:, ['Nb_char', 'Nb_word', 'Avg_word_len', 'Nb_Exclam', 'Nb_Upper', 'Nb_Hashtag', 'Nb_At', 'Nb_link', 'Hashtag', 'At', 'Link', 'topic']]
Y = train_df.loc[:, 'target']

# Divide dataset Train set & Test set 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 0, stratify = Y)


# In[38]:


cont_features = ['Nb_char', 'Nb_word', 'Avg_word_len', 'Nb_Exclam', 'Nb_Upper', 'Nb_Hashtag', 'Nb_At', 'Nb_link']
cat_features = ['Hashtag', 'At', 'Link', 'topic']

pipelines = [
    ('continu', 
     Pipeline(steps = [('scaler', StandardScaler(with_mean = True))]), 
     cont_features),
    ('categorical', 
     Pipeline(steps = [('encoder', OneHotEncoder(drop = 'first'))]), 
     cat_features)]

pips = ColumnTransformer(transformers = pipelines)

X_train = pips.fit_transform(X_train)
X_test = pips.transform(X_test)


# In[39]:


names = [
    "Logistic regression",
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
    "SGDC",
    "Gradient Boosting",
    "XGBoost",
    "Extra Trees"
]

classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(10),
    SVC(kernel="linear"),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    SGDClassifier(loss="log", penalty="elasticnet"),
    GradientBoostingClassifier(),
    XGBClassifier(),
    ExtraTreesClassifier()
]

Bilan = []

for i in range(3,len(classifiers)) :
    model = classifiers[i]
    model.fit(X_train, Y_train)

    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)

    Bilan.append({'Model'           : names[i],
                  'Accuracy train'  : accuracy_score(Y_train, Y_train_pred),
                  'Accuracy test'   : accuracy_score(Y_test, Y_test_pred),
                  'Precision train' : precision_score(Y_train, Y_train_pred),
                  'Precision test'  : precision_score(Y_test, Y_test_pred),
                  'Recall train'    : recall_score(Y_train, Y_train_pred),
                  'Recall test'     : recall_score(Y_test, Y_test_pred),
                  'f1 score train'  : f1_score(Y_train, Y_train_pred),
                  'f1 score test'   : f1_score(Y_test, Y_test_pred), 
                  'ROC score train' : roc_auc_score(Y_train, Y_train_pred),
                  'ROC score test'  : roc_auc_score(Y_test, Y_test_pred)
                  }
                 )


# In[40]:


pd.DataFrame(Bilan).round(3)


# ## 5.2. Classifiers - TF-IDF and BoW
# 
# We then tried to use the same ML algorithms on the text itself. However, to do so, text had to be transformed into readable variables. There are two main ways that allow to transform text into variables. 
# 
# **Bag of Words (BoW)** : Using BoW, we transform a textual information into a representation that contains the occurrence of words within a document. It removes a lot of information from text, such as grammar, words order, meaning to keep only the word counts. This loss of information leads to the name of the method : Bag of Words as all words are putting together into a bag, regardless of their order and grammatical aspects. 
# 
# **Term Frequency and Inverse Document Frequency (TF-IDF)** : More complex that BoW, TF-IDF is a scoring / weighting system traditionnally used in the natural language processing (NLP). It combines information from terms and documents to quantify a term relevance in a document considering its occurrence into a complete corpus. 
# 

# In[41]:


Y = train_df.loc[:, 'target']

# Divide dataset Train set & Test set 
X_train, X_test, Y_train, Y_test = train_test_split(train_df['desc_lemma_text'], Y, test_size = 0.1, random_state = 0, stratify = Y)


# In[42]:


# TF-IDF 
vectorizer = TfidfVectorizer(stop_words='english', use_idf = 1, smooth_idf = 1, sublinear_tf = 1)
X_tfidf_train = vectorizer.fit_transform(X_train).toarray()
X_tfidf_train = pd.DataFrame(X_tfidf_train, columns = vectorizer.get_feature_names())
X_tfidf_test = vectorizer.transform(X_test).toarray()
X_tfidf_test = pd.DataFrame(X_tfidf_test,columns = vectorizer.get_feature_names())

# BoW
vectorizer = CountVectorizer(stop_words='english')
X_BoW_train = vectorizer.fit_transform(X_train).toarray()
X_BoW_train = pd.DataFrame(X_BoW_train, columns = vectorizer.get_feature_names())
X_BoW_test = vectorizer.transform(X_test).toarray()
X_BoW_test = pd.DataFrame(X_BoW_test,columns = vectorizer.get_feature_names())


# In[43]:


names = [
    "Logistic regression",
    "Nearest Neighbors",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "SGDC",
    "XGBoost"]

classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(10),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    SGDClassifier(loss="log", penalty="elasticnet"),
    XGBClassifier()
]

for i in range(0,len(classifiers)) :
    model = classifiers[i]
    model.fit(X_tfidf_train, Y_train)

    Y_train_pred = model.predict(X_tfidf_train)
    Y_test_pred = model.predict(X_tfidf_test)
    print (names[i])
    Bilan.append({'Model'           : 'TF-IFD ' + names[i],
                  'Accuracy train'  : accuracy_score(Y_train, Y_train_pred),
                  'Accuracy test'   : accuracy_score(Y_test, Y_test_pred),
                  'Precision train' : precision_score(Y_train, Y_train_pred),
                  'Precision test'  : precision_score(Y_test, Y_test_pred),
                  'Recall train'    : recall_score(Y_train, Y_train_pred),
                  'Recall test'     : recall_score(Y_test, Y_test_pred),
                  'f1 score train'  : f1_score(Y_train, Y_train_pred),
                  'f1 score test'   : f1_score(Y_test, Y_test_pred), 
                  'ROC score train' : roc_auc_score(Y_train, Y_train_pred),
                  'ROC score test'  : roc_auc_score(Y_test, Y_test_pred)
                  }
                 )


# In[44]:


pd.DataFrame(Bilan).round(3)


# In[45]:


for i in range(0,len(classifiers)) :
    model = classifiers[i]
    model.fit(X_BoW_train, Y_train)

    Y_train_pred = model.predict(X_BoW_train)
    Y_test_pred = model.predict(X_BoW_test)

    Bilan.append({'Model'           : 'BoW ' + names[i],
                  'Accuracy train'  : accuracy_score(Y_train, Y_train_pred),
                  'Accuracy test'   : accuracy_score(Y_test, Y_test_pred),
                  'Precision train' : precision_score(Y_train, Y_train_pred),
                  'Precision test'  : precision_score(Y_test, Y_test_pred),
                  'Recall train'    : recall_score(Y_train, Y_train_pred),
                  'Recall test'     : recall_score(Y_test, Y_test_pred),
                  'f1 score train'  : f1_score(Y_train, Y_train_pred),
                  'f1 score test'   : f1_score(Y_test, Y_test_pred), 
                  'ROC score train' : roc_auc_score(Y_train, Y_train_pred),
                  'ROC score test'  : roc_auc_score(Y_test, Y_test_pred)
                  }
                 )


# In[46]:


pd.DataFrame(Bilan).round(3)


# ## 5.3. Neural networks
# 
# Dealing with natural texts, neural networks may offer interesting performances as they allow to deal with the text sequentiality. 

# ### 5.3.1. Tokenization and Spliting
# 
# The tokenization is simply the process of splitting a text into smaller units (usually words) that are called tokens. 

# In[47]:


tokenizer = tf.keras.preprocessing.text.Tokenizer() 
tokenizer.fit_on_texts(train_df["cleaned"])
vocab_length = len(tokenizer.word_index) + 1


# In[48]:


train_df["tweet_encoded"] = tokenizer.texts_to_sequences(train_df["cleaned"])
pads = tf.keras.preprocessing.sequence.pad_sequences(train_df['tweet_encoded'], padding="post")


# In[49]:


Y = train_df['target']
# Divide dataset Train set & Test set 
X_train, X_test, Y_train, Y_test = train_test_split(pads, Y, test_size = 0.1, random_state = 0, stratify = Y)
X_train


# In[50]:


# Creation of tensorflow tensors from train and validation sets
train_ts = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
test_ts = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
# Shuffling and batching
train_ts = train_ts.shuffle(len(train_ts)).batch(128)
test_ts = test_ts.shuffle(len(test_ts)).batch(128)


# ### 5.3.2. Recurrent Neural Network (RNN)
# 
# RNN are a reference algorithm for sequential data thanks to its internal memory (it remembers its input). For this reason, RNN is widely used for voice / speech treatement. Its memory allows RNN to predict what is coming next and to has a better understanding of a sequence and of a context compared to other kind of algorithms. 
# 
# RNN is usually opposed to feed-forward neural networks. These last ones ony allow information to move forward from one layer to the next one until the output layer. This constraint does not allow FFNN to remember previous information. 
# In RNN, information can circulate through a loop in which the current decision can be informed by the previously received input. Thus, a RNN has 2 simultaneous inputs : the present and the recent / immediate past which is crucial which dealing with sequential information such as text. 
# 
# RNN have 2 main issues, both being related to the notion of gradient. The gradient is a partial derivative that allows to measure the change of an output due to a small change in the input. Said differently, a gradient measures the change in the network weights after a change in error. 
# - Exploding gradients : Too much importance is put on weights. 
# - Vanishing gradients : The gradient becomes too small leading to a model unable to learn or learning very slowly. 

# In[51]:


print('Maximum lenght for a tweet :', max(train_df['Nb_word']))
plt.boxplot(train_df['Nb_word']);


# As the length of the tweets in the database remained reasonable, we can begin by a simple recurrent neural network (RNN) structure.
# 
# How to parametrize a RNN ? 
# - First layer - Embedding : The parameters of this layer are the following. The input dimension is the size of the vocabulary / corpus. The input length is the number of entries (dimension of the dataset in input). The output dimension is the size of the vector space in which words will be embedded (usually between 32 and 100 - this parameter could be optimized)
# - Second layer - Simple RNN : We include a number of units / neurons. 
# - Additional layers - ReLu and Dropouts :  ReLu is useful for the backpropagation of the error and can be very useful associated with sigmoidal functions as it prevents the vanishing gradient problem. Dropouts randomly deactivates a part of the neurons to prevent overfitting (by nullifying their contribution to the output)
# - Final layer - Output : One neuron with sigmoid activation function (to predict a binary endpoint)

# In[52]:


model_sRNN = tf.keras.models.Sequential([
  tf.keras.layers.Embedding(input_dim = pads.max() + 1, 
                            output_dim = 32,
                            input_length = (X_train.shape[-1])),
  tf.keras.layers.SimpleRNN(units = 16, 
                            return_sequences = False), 
  tf.keras.layers.Dense(64, activation = 'relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1, activation = "sigmoid") 
])

display(model_sRNN.summary())
#tf.keras.utils.plot_model(model_sRNN, show_shapes = True)


# We used Adam optimizer (Adaptative Moment estimation) which replaced the classci stochastic gradient descent and which is more specificatlly dedicated to NLP. 

# In[53]:


model_sRNN.compile(optimizer = tf.keras.optimizers.Adam(1e-4),
              loss = tf.keras.losses.BinaryCrossentropy(),
              metrics = ['accuracy', 
                         tfa.metrics.F1Score(num_classes = 1, average = 'micro', threshold = 0.5)])


# In[54]:


model_sRNN.fit(
    train_ts,
    validation_data = test_ts,
    epochs = 15)


# In[55]:


pio.renderers.default = "notebook"

history = model_sRNN.history.history
fig = make_subplots(rows = 1, cols = 3, subplot_titles = ("Loss", "Accuracy", "F1-score"))

fig.add_trace(go.Scatter(y = history["loss"],
                    mode = 'lines',
                    name = 'loss',
                    legendgroup = "Loss",
                    line_color = "pink"),
              row = 1,
              col = 1)
fig.add_trace(go.Scatter(y = history["val_loss"],
                    mode = 'lines',
                    name = 'val_loss',
                    legendgroup = "Loss",
                    line_color = "red"),
              row = 1,
              col = 1)

fig.add_trace(go.Scatter(y = history["accuracy"],
                    mode = 'lines',
                    name = 'accuracy',
                    legendgroup = "Accuracy",
                    line_color = "lightblue"),
              row = 1,
              col = 2)
fig.add_trace(go.Scatter(y = history["val_accuracy"],
                    mode = 'lines',
                    name = 'val_accuracy',
                    legendgroup = "Accuracy",
                    line_color = "blue"),
              row = 1,
              col = 2)

fig.add_trace(go.Scatter(y = history["f1_score"],
                    mode = 'lines',
                    name = 'f1_score',
                    legendgroup = "F1-score",
                    line_color = "lightgreen"),
              row = 1,
              col = 3)
fig.add_trace(go.Scatter(y = history["val_f1_score"],
                    mode = 'lines',
                    name = 'val_f1_score',
                    legendgroup = "F1-score",
                    line_color = "green"),
              row = 1,
              col = 3)

fig.update_layout(title_text = "Simple RNN model performances", width = 1500,
                  yaxis2_range = [0, 1], yaxis3_range = [0, 1])
fig.show()


# The sRNN begins to overfit after 7 epochs. We used this parameter for its final training

# In[56]:


model_sRNN = tf.keras.models.Sequential([
  tf.keras.layers.Embedding(input_dim = pads.max() + 1, 
                            output_dim = 32,
                            input_length = (X_train.shape[-1])),
  tf.keras.layers.SimpleRNN(units = 16, 
                            return_sequences = False), 
  tf.keras.layers.Dense(64, activation = 'relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1, activation = "sigmoid") 
])
model_sRNN.compile(optimizer = tf.keras.optimizers.Adam(1e-4),
              loss = tf.keras.losses.BinaryCrossentropy(),
              metrics = ['accuracy', 
                         tfa.metrics.F1Score(num_classes = 1, average = 'micro', threshold = 0.5)])
model_sRNN.fit(
    train_ts,
    validation_data = test_ts,
    epochs = 7)


# In[57]:


Bilan.append({'Model'           : 'sRNN',
                  'Accuracy train'  : accuracy_score(Y_train, np.round(model_sRNN(X_train))),
                  'Accuracy test'   : accuracy_score(Y_test, np.round(model_sRNN(X_test))),
                  'Precision train' : precision_score(Y_train, np.round(model_sRNN(X_train))),
                  'Precision test'  : precision_score(Y_test,  np.round(model_sRNN(X_test))),
                  'Recall train'    : recall_score(Y_train, np.round(model_sRNN(X_train))),
                  'Recall test'     : recall_score(Y_test,  np.round(model_sRNN(X_test))),
                  'f1 score train'  : f1_score(Y_train, np.round(model_sRNN(X_train))),
                  'f1 score test'   : f1_score(Y_test,  np.round(model_sRNN(X_test))), 
                  'ROC score train' : roc_auc_score(Y_train, np.round(model_sRNN(X_train))),
                  'ROC score test'  : roc_auc_score(Y_test,  np.round(model_sRNN(X_test)))
                  }
                 )


# In[58]:


round(pd.DataFrame(Bilan),3)


# ### 5.3.3. Long Short-Term Memory (LSTM) RNN
# 
# LSTM is an extension of the RNN models and has been introduced to solve the 2 gradient issues of RNN. LSTM can be seen as a memory extension. As we have seen, RNN has 2 inputs : the present and the immediate past. However, when we have information with a long sequence, this king of short term memory may not be sufficient. 
# With the introduction of "gates" (output gate, forget gate, input/update gate), the LSTM neurons will be able to store (or not) and then to delete (or not) the information based on this information relevance / importance (weights).   
# 

# In[59]:


model_LSTM = tf.keras.models.Sequential([
  tf.keras.layers.Embedding(input_dim = pads.max() + 1, 
                            output_dim = 32,
                            input_length = (X_train.shape[-1])),
  tf.keras.layers.LSTM(units=16, return_sequences=False), # returns the last output
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1, activation="sigmoid") # the prediction layer
])

display(model_LSTM.summary())
#tf.keras.utils.plot_model(model_LSTM, show_shapes = True)


# In[60]:


model_LSTM.compile(optimizer = tf.keras.optimizers.Adam(1e-4),
              loss = tf.keras.losses.BinaryCrossentropy(),
              metrics = ['accuracy', 
                         tfa.metrics.F1Score(num_classes = 1, average = 'micro', threshold = 0.5)])


# In[61]:


model_LSTM.fit(
    train_ts,
    validation_data = test_ts,
    epochs = 15)


# In[62]:


pio.renderers.default = "notebook"

history = model_LSTM.history.history
fig = make_subplots(rows = 1, cols = 3, subplot_titles = ("Loss", "Accuracy", "F1-score"))

fig.add_trace(go.Scatter(y = history["loss"],
                    mode = 'lines',
                    name = 'loss',
                    legendgroup = "Loss",
                    line_color = "pink"),
              row = 1,
              col = 1)
fig.add_trace(go.Scatter(y = history["val_loss"],
                    mode = 'lines',
                    name = 'val_loss',
                    legendgroup = "Loss",
                    line_color = "red"),
              row = 1,
              col = 1)

fig.add_trace(go.Scatter(y = history["accuracy"],
                    mode = 'lines',
                    name = 'accuracy',
                    legendgroup = "Accuracy",
                    line_color = "lightblue"),
              row = 1,
              col = 2)
fig.add_trace(go.Scatter(y = history["val_accuracy"],
                    mode = 'lines',
                    name = 'val_accuracy',
                    legendgroup = "Accuracy",
                    line_color = "blue"),
              row = 1,
              col = 2)

fig.add_trace(go.Scatter(y = history["f1_score"],
                    mode = 'lines',
                    name = 'f1_score',
                    legendgroup = "F1-score",
                    line_color = "lightgreen"),
              row = 1,
              col = 3)
fig.add_trace(go.Scatter(y = history["val_f1_score"],
                    mode = 'lines',
                    name = 'val_f1_score',
                    legendgroup = "F1-score",
                    line_color = "green"),
              row = 1,
              col = 3)

fig.update_layout(title_text = "LSTM model performances", width = 1500,
                  yaxis2_range = [0, 1], yaxis3_range = [0, 1])
fig.show()


# The model begins to overfit after 8 epochs. We used this parameter for its final training

# In[63]:


model_LSTM = tf.keras.models.Sequential([
  tf.keras.layers.Embedding(input_dim = pads.max() + 1, 
                            output_dim = 32,
                            input_length = (X_train.shape[-1])),
  tf.keras.layers.LSTM(units=16, return_sequences=False), # returns the last output
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1, activation="sigmoid") # the prediction layer
])
model_LSTM.compile(optimizer = tf.keras.optimizers.Adam(1e-4),
              loss = tf.keras.losses.BinaryCrossentropy(),
              metrics = ['accuracy', 
                         tfa.metrics.F1Score(num_classes = 1, average = 'micro', threshold = 0.5)])
model_LSTM.fit(
    train_ts,
    validation_data = test_ts,
    epochs = 10)


# In[64]:


Bilan.append({'Model'           : 'LSTM',
                  'Accuracy train'  : accuracy_score(Y_train, np.round(model_LSTM(X_train))),
                  'Accuracy test'   : accuracy_score(Y_test, np.round(model_LSTM(X_test))),
                  'Precision train' : precision_score(Y_train, np.round(model_LSTM(X_train))),
                  'Precision test'  : precision_score(Y_test,  np.round(model_LSTM(X_test))),
                  'Recall train'    : recall_score(Y_train, np.round(model_LSTM(X_train))),
                  'Recall test'     : recall_score(Y_test,  np.round(model_LSTM(X_test))),
                  'f1 score train'  : f1_score(Y_train, np.round(model_LSTM(X_train))),
                  'f1 score test'   : f1_score(Y_test,  np.round(model_LSTM(X_test))), 
                  'ROC score train' : roc_auc_score(Y_train, np.round(model_LSTM(X_train))),
                  'ROC score test'  : roc_auc_score(Y_test,  np.round(model_LSTM(X_test)))
                  }
                 )


# In[65]:


round(pd.DataFrame(Bilan),3)


# ### 5.3.4. Bidirectional LSTM
# 
# The bLSTM has been used primarily on natural language processing. Its bidirectional characteristic allows the input to flow in both direction and the decision is made according to the 2 flows at a given time. the bLSTM is simply the addition of a second LSTM layer in the reverse direction and output from the 2 layers are combined to lead to a decision. 

# In[66]:


model_bLSTM = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim = pads.max() + 1, 
                            output_dim = 32,
                            input_length = (X_train.shape[-1])),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16,return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
    tf.keras.layers.Dense(18,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(9,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1,activation='sigmoid')])
model_bLSTM.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 
                         tfa.metrics.F1Score(num_classes = 1, average = 'micro', threshold = 0.5)])
model_bLSTM.summary()


# In[67]:


# Entrainement du modèle 
history = model_bLSTM.fit(train_ts,
    validation_data = test_ts,
    epochs = 20)


# In[68]:


pio.renderers.default = "notebook"

history = model_bLSTM.history.history
fig = make_subplots(rows = 1, cols = 3, subplot_titles = ("Loss", "Accuracy", "F1-score"))

fig.add_trace(go.Scatter(y = history["loss"],
                    mode = 'lines',
                    name = 'loss',
                    legendgroup = "Loss",
                    line_color = "pink"),
              row = 1,
              col = 1)
fig.add_trace(go.Scatter(y = history["val_loss"],
                    mode = 'lines',
                    name = 'val_loss',
                    legendgroup = "Loss",
                    line_color = "red"),
              row = 1,
              col = 1)

fig.add_trace(go.Scatter(y = history["accuracy"],
                    mode = 'lines',
                    name = 'accuracy',
                    legendgroup = "Accuracy",
                    line_color = "lightblue"),
              row = 1,
              col = 2)
fig.add_trace(go.Scatter(y = history["val_accuracy"],
                    mode = 'lines',
                    name = 'val_accuracy',
                    legendgroup = "Accuracy",
                    line_color = "blue"),
              row = 1,
              col = 2)

fig.add_trace(go.Scatter(y = history["f1_score"],
                    mode = 'lines',
                    name = 'f1_score',
                    legendgroup = "F1-score",
                    line_color = "lightgreen"),
              row = 1,
              col = 3)
fig.add_trace(go.Scatter(y = history["val_f1_score"],
                    mode = 'lines',
                    name = 'val_f1_score',
                    legendgroup = "F1-score",
                    line_color = "green"),
              row = 1,
              col = 3)

fig.update_layout(title_text = "bLSTM model performances", width = 1500,
                  yaxis2_range = [0, 1], yaxis3_range = [0, 1])
fig.show()


# In[69]:


model_bLSTM = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim = pads.max() + 1, 
                            output_dim = 32,
                            input_length = (X_train.shape[-1])),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16,return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
    tf.keras.layers.Dense(18,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(9,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1,activation='sigmoid')])
model_bLSTM.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 
                         tfa.metrics.F1Score(num_classes = 1, average = 'micro', threshold = 0.5)])

model_bLSTM.fit(
    train_ts,
    validation_data = test_ts,
    epochs = 10)


# In[70]:


Bilan.append({'Model'           : 'bLSTM',
                  'Accuracy train'  : accuracy_score(Y_train, np.round(model_bLSTM(X_train))),
                  'Accuracy test'   : accuracy_score(Y_test, np.round(model_bLSTM(X_test))),
                  'Precision train' : precision_score(Y_train, np.round(model_bLSTM(X_train))),
                  'Precision test'  : precision_score(Y_test,  np.round(model_bLSTM(X_test))),
                  'Recall train'    : recall_score(Y_train, np.round(model_bLSTM(X_train))),
                  'Recall test'     : recall_score(Y_test,  np.round(model_bLSTM(X_test))),
                  'f1 score train'  : f1_score(Y_train, np.round(model_bLSTM(X_train))),
                  'f1 score test'   : f1_score(Y_test,  np.round(model_bLSTM(X_test))), 
                  'ROC score train' : roc_auc_score(Y_train, np.round(model_bLSTM(X_train))),
                  'ROC score test'  : roc_auc_score(Y_test,  np.round(model_bLSTM(X_test)))
                  }
                 )


# In[71]:


round(pd.DataFrame(Bilan),3)


# ### 5.3.5. Transfer learning - Universal sentence encoder
# 
# In this final algorithm, we will use a pre-trained embedding layer, the Universal Sentence Encoder, specially conceived to generate sentence embeddings. 

# In[73]:


module_url = 'https://tfhub.dev/google/universal-sentence-encoder-large/5'


# In[ ]:


model_USE = tf.keras.models.Sequential([
  hub.KerasLayer(module_url, trainable=True, name='USE_embedding'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1, activation="sigmoid") # the prediction layer
])


# In[75]:


model_USE.compile(optimizer = tf.keras.optimizers.Adam(1e-4),
              loss = tf.keras.losses.BinaryCrossentropy(),
              metrics = ['accuracy', 
                         tfa.metrics.F1Score(num_classes = 1, average = 'micro', threshold = 0.5)])


# In[76]:


X_train, X_test, Y_train, Y_test = train_test_split(train_df['text'], Y, test_size = 0.1, random_state = 0, stratify = Y)


# In[77]:


# Creation of tensorflow tensors from train and validation sets
train_ts = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
test_ts = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

# Shuffling and batching
train_ts = train_ts.shuffle(len(train_ts)).batch(128)
test_ts = test_ts.shuffle(len(test_ts)).batch(128)


# In[ ]:


model_USE.fit(
    train_ts,
    validation_data = test_ts, 
    epochs = 20)


# In[79]:


pio.renderers.default = "notebook"

history = model_USE.history.history
fig = make_subplots(rows = 1, cols = 3, subplot_titles = ("Loss", "Accuracy", "F1-score"))

fig.add_trace(go.Scatter(y = history["loss"],
                    mode = 'lines',
                    name = 'loss',
                    legendgroup = "Loss",
                    line_color = "pink"),
              row = 1,
              col = 1)
fig.add_trace(go.Scatter(y = history["val_loss"],
                    mode = 'lines',
                    name = 'val_loss',
                    legendgroup = "Loss",
                    line_color = "red"),
              row = 1,
              col = 1)

fig.add_trace(go.Scatter(y = history["accuracy"],
                    mode = 'lines',
                    name = 'accuracy',
                    legendgroup = "Accuracy",
                    line_color = "lightblue"),
              row = 1,
              col = 2)
fig.add_trace(go.Scatter(y = history["val_accuracy"],
                    mode = 'lines',
                    name = 'val_accuracy',
                    legendgroup = "Accuracy",
                    line_color = "blue"),
              row = 1,
              col = 2)

fig.add_trace(go.Scatter(y = history["f1_score"],
                    mode = 'lines',
                    name = 'f1_score',
                    legendgroup = "F1-score",
                    line_color = "lightgreen"),
              row = 1,
              col = 3)
fig.add_trace(go.Scatter(y = history["val_f1_score"],
                    mode = 'lines',
                    name = 'val_f1_score',
                    legendgroup = "F1-score",
                    line_color = "green"),
              row = 1,
              col = 3)

fig.update_layout(title_text = "USE model performances", width = 1500,
                  yaxis2_range = [0, 1], yaxis3_range = [0, 1])
fig.show()


# In[ ]:


model_USE = tf.keras.models.Sequential([
  hub.KerasLayer(module_url, trainable=True, name='USE_embedding'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1, activation="sigmoid") # the prediction layer
])

model_USE.compile(optimizer = tf.keras.optimizers.Adam(1e-4),
              loss = tf.keras.losses.BinaryCrossentropy(),
              metrics = ['accuracy', 
                         tfa.metrics.F1Score(num_classes = 1, average = 'micro', threshold = 0.5)])

model_USE.fit(
    train_ts,
    validation_data = test_ts, 
    epochs = 5)


# In[81]:


X_train_pred = model_USE.predict(np.array(X_train, dtype=object)[:, np.newaxis], batch_size=32)
X_test_pred = model_USE.predict(np.array(X_test, dtype=object)[:, np.newaxis], batch_size=32)


# In[82]:


Bilan.append({'Model'           : 'USE',
                  'Accuracy train'  : accuracy_score(Y_train, np.round(X_train_pred)),
                  'Accuracy test'   : accuracy_score(Y_test, np.round(X_test_pred)),
                  'Precision train' : precision_score(Y_train, np.round(X_train_pred)),
                  'Precision test'  : precision_score(Y_test,  np.round(X_test_pred)),
                  'Recall train'    : recall_score(Y_train, np.round(X_train_pred)),
                  'Recall test'     : recall_score(Y_test,  np.round(X_test_pred)),
                  'f1 score train'  : f1_score(Y_train, np.round(X_train_pred)),
                  'f1 score test'   : f1_score(Y_test,  np.round(X_test_pred)), 
                  'ROC score train' : roc_auc_score(Y_train, np.round(X_train_pred)),
                  'ROC score test'  : roc_auc_score(Y_test,  np.round(X_test_pred))
                  }
                 )
round(pd.DataFrame(Bilan),3)

