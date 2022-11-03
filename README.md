# 1. Project presentation : Natural Language Processing with Disaster Tweets

This project is one of the getting started challenges offered by [Kaggle](https://www.kaggle.com/c/nlp-getting-started). 
As an instant way of communication, Twitter has become an important channel in times of emergency. It enables people to announce an emergency in real-time. Because of this immediacy, a growing numhber of agencies are interested in automatically monitoring Twitter.

----

# 2. Objective

The objective is to build a machine learning model that predicts which Tweets are about real disasters and which one’s aren’t. 

----

3. Comment procéder ?

Pré-requis

Les librairies suivantes sont nécessaires :

    re
    unidecode
    bs4
    requests
    math
    pandas
    json
    plotly
    datetime
    os
    logging
    scrapy
    folium
    pyproj
    matplotlib
    colour
    turtle
    IPython

Les fichiers

Les notebooks peuvent s'utiliser les uns à la suite des autres ou indépendamment puisque les données générées par chaque notebook sont fournies dans les fichiers csv.

    Etape 1 - Infos des hotels.ipynb fournit le scraping de Booking.com selon 2 méthodes :
        Utilisation de SCRAPY -> table_hotels_scrapy.csv
        Utilisation de BEAUTIFULSOUP -> table_hotels_BS.csv
    Etape 2 - Données Météo.ipynb prend en input table_hotels_scrapy.csv ou table_hotels_BS.csv et fournit les données météorologiques
        Coordonnées GPS des villes : table_villes_coord.csv
        Météo d'aujourd'hui et des 7 prochains jours : data_meteo.csv
    Etape 3 - Cartographies.ipynb prend en input les 3 fichiers csv (hotels, coord et meteo)
    Etape 4 - ETL.ipynb prend en input les 3 fichiers csv (hotels, coord et meteo)

## Dataset

Kaggle provides a dataset of 10 000 tweets that were hand classified as disaster-related or not. In the train dataset, around 43% are disaster-related. 

---

4. Overview des principaux résultats

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

---- 

# 5. Informations

## Outils

Les notebooks ont été développés avec Visual Studio Code. La partie ETL fonctionne avec :

    Un compte AWS (payant)
    PGAdmin

## Auteurs & contributeurs

Auteur :

    Helene alias @Bebock

La dream team :

    Henri alias @HenriPuntous
    Jean alias @Chedeta
    Nicolas alias @NBridelance

