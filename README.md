# 1. Project presentation : Natural Language Processing with Disaster Tweets

This project is one of the getting started challenges offered by [Kaggle](https://www.kaggle.com/c/nlp-getting-started). 
As an instant way of communication, Twitter has become an important channel in times of emergency. It enables people to announce an emergency in real-time. Because of this immediacy, a growing numhber of agencies are interested in automatically monitoring Twitter.

----

# 2. Objective

The objective is to build a machine learning model that predicts which Tweets are about real disasters and which one’s aren’t. 

----

# 3. How to procede ?

## Requirements

The following librairies are required :

    re
    unidecode
    spellchecker
    contractions
    string
    math
    pandas
    numpy
    plotly
    matplotlib
    seaborn
    spacy
    nltk
    gensim
    pyLDAvis
    wordcloud
    sklearn
    xgboost
    tensorflow

## Dataset

Kaggle provides a dataset of 10 000 tweets that were hand classified as disaster-related or not. In the train dataset, around 43% are disaster-related. 

---

# 4. Overview of the main results

## Exploratory Descriptive Analysis

First of all, we extracted some quantitative features, not directly linked to the contents. Doing so, we tried to extract some additional information not available through text / content analysis.
- Lenght of the tweet measured by the number of characters and by the number of words
- Average lenght of words
- Number of exclamation marks
- Number of uppercase letters
- Number and presence of #
- Number and presence of @
- Number and presence of urls

Statistically speaking, all these characteristics were related to the target variable (disaster tweet or not). However, the sample size of the dataset provides too much statistical power so we focused on graphical explorations. 
It appeared that the disaster-related tweets contain longer words in average, less uppercases, more #, much less @, and less urls in mean but the disaster-related tweets are in proportion more prone to contain at least one url. 

![image](https://user-images.githubusercontent.com/38078432/199684886-fa83a42a-578f-4fa9-8b0d-9e2ca9825776.png)

## Content description 

The tweets content has been described after pre-processing and lemmatization with : 
- Top bigrams according to the target variable (disaster-related or not)

![image](https://user-images.githubusercontent.com/38078432/199688800-c420992d-8514-4831-85d8-d46c41d7f43e.png)

- Wordclouds according to the target variable (disaster-related or not)

![image](https://user-images.githubusercontent.com/38078432/200079057-fc9bd898-c497-4e90-9a4e-2d85df54002d.png)

## Sentiment analysis

Disaster tweets appeared less associated with neutral or positive sentiments. 

![image](https://user-images.githubusercontent.com/38078432/199689331-ae2eb7dc-4271-49e7-b795-351a7cc0a79e.png)

However, the sentiment analysis allowed us to detect some questionning coding in the dataset. Indeed, displaying the tweets coded as disaster-related and categorized as positive by the sentiment analysis, we can read for exemple the following tweets : 
- "my favorite lady came to our volunteer meeting hopefully joining her youth collision and i am excite" 
- "ok peace I hope I fall off a cliff along with my dignity"
- ":) well I think that sounds like a fine plan where little derailment is possible so I applaud you :)"
... and they do not refer to disasters. It might indicate some confusing labels in the training dataset. 

## Prediction models : How to automatically detect a disaster-related tweet ?

We tried 3 main approaches : 
- Using classical Machine learning models on the previously extracte features 
- Using classical Machine learning models on recoded text through TF-IDF or BoW
- Using neural networks

The results were the following : 
![image](https://user-images.githubusercontent.com/38078432/202900172-2100d79c-4a29-4c00-a7d2-9cc3e4f50ec0.png)

It appeared that a pre-trained embedding neural netword (Universal Sentence Encoder) had the best performances in terms of ROC score, f1 score and accuracy. However, classical Machine Leanrning models performed on Bag of Words (Neural Net and SGDC) performed at a similar level. 

## Perspectives 

The best models may now be improved thanks to fine tuning of hyperparameters. 

---- 

# 5. Informations

## Tools

The notebook has been developed with Visual Studio Code.

The USE is available [here](https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder)

## Authors & contributors

Author :

    Helene alias @Bebock

The dream team :

    Henri alias @HenriPuntous
    Jean alias @Chedeta
    Nicolas alias @NBridelance

