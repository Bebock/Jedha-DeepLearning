# Natural Language Processing with Disaster Tweets

## Context 

This project is one of the getting started challenges offered by [Kaggle](https://www.kaggle.com/c/nlp-getting-started). 
As an instant way of communication, Twitter has become an important channel in times of emergency. It enables people to announce an emergency in real-time. Because of this immediacy, a growing numhber of agencies are interested in automatically monitoring Twitter.

## Objective

The objective is to build a machine learning model that predicts which Tweets are about real disasters and which one’s aren’t. 

## Dataset

Kaggle provides a dataset of 10 000 tweets that were hand classified as disaster-related or not. In the train dataset, around 43% are disaster-related. 

---

# Exploratory Descriptive Analysis

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


