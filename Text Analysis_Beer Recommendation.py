#!/usr/bin/env python
# coding: utf-8

# ## Building a Crowdsourced Recommendation System

# In[1]:


# import module and upload file
import pandas as pd
import numpy as np
#from google.colab import files
#df = files.upload()


# ### High level description: 
# 
# The objective of this project is to create the building blocks of a crowdsourced recommendation system. This recommendation system should accept user inputs about desired attributes of a product and come up with 3 recommendations.
# 
# Obtain reviews of craft beer from beeradvocate.com. We would suggest using the following link, which shows the top 250 beers sorted by ratings: https://www.beeradvocate.com/beer/top-rated/
# 
# The nice feature of the above link is that it is a single-page listing of 250 top-rated beers (avoids the pagination feature, which we need in cases where listings go on for many pages). The way beeradvocate.com organizes reviews is that it provides about 25 reviews per page. The output file should have 3 columns: product_name, product_review, and user_rating.

# ### Task A. Extract about 5-6k reviews. 

# In[2]:


beer = pd.read_csv('cleaned_beer.csv',index_col=0)
beer.head()


# In[3]:


#import module
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
stop = stopwords.words('english')

tokenizer = RegexpTokenizer(r'\w+')

#tokenize the review
beer['tokenized_review'] = beer['cleaned_review'].astype('str').apply(lambda x: tokenizer.tokenize(x))


# In[4]:


beer_name = list(beer['beer'].unique())
len(beer_name)


# In[5]:


len(beer)


# ### Task B. Specify 3 attributes in a product.
# 
# Assume that a customer, who will be using this recommender system, has specified 3 attributes in a product. E.g., one website describes multiple attributes of beer:
# https://www.dummies.com/food-drink/drinks/beer/beer-for-dummies-cheat-sheet/
# 
# - Aggressive (Boldly assertive aroma and/or taste) 
# - Balanced: Malt and hops in similar proportions; equal representation of malt sweetness and hop bitterness in the flavor — especially at the finish
# - Complex: Multidimensional; many flavors and sensations on the palate
# - Crisp: Highly carbonated; effervescent
# - Fruity: Flavors reminiscent of various fruits
# - Hoppy: Herbal, earthy, spicy, or citric aromas and flavors of hops
# - Malty: Grainy, caramel-like; can be sweet or dry
# - Robust: Rich and full-bodied
# 
# A word frequency analysis of beer reviews may be a better way to find important attributes.
# 
# Assume that a customer has specified three attributes of the product as being important to him or her.

# In[6]:


import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')


# In[7]:


tokens = beer['cleaned_review'].astype('str').apply(nltk.word_tokenize)
from itertools import chain
words = list(chain(*tokens))

from collections import Counter
attribute = ['aggressive', 'balanced', 'complex', 'crisp', 'fruity', 'hoppy', 'malty', 'robust']
words = [word.lower() for word in words]
count_tuple = Counter(words)

for char, count in count_tuple.items():
    if char in attribute:
        print(char, count)
# balanced, complex, fruity


# ### Task C. Similarity Analysis.
# 
# Perform a similarity analysis using cosine similarity (without word embeddings) with the 3 attributes specified by the customer and the reviews. From the output file, calculate the average similarity between each product and the preferred attributes.
# 
# For similarity analysis, use cosine similarity with bag of words. The script should accept as input a file with the product attributes, and calculate similarity scores (between 0 and 1) between these attributes and each review. That is, the output file should have 3 columns – product_name (for each product, the product_name will repeat as many times as there are reviews of the product), product_review and similarity_score. 

# In[8]:


# three attributes specified by the customer and the reviews: balanced, complex, fruity
beer['tokenized_review'] = beer['tokenized_review'].apply(lambda x: [word.lower() for word in x if word not in stop])
beer.head()


# In[9]:


beer.isna().sum()


# In[10]:


# remove null value
beer.dropna(inplace=True)


# In[11]:


#Calculate cosine similarity using Bag-of-Words

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def cos(x):
    documents =[x, 'fruity complex balanced']
    count_vectorizer = CountVectorizer(stop_words='english')
    sparse_matrix = count_vectorizer.fit_transform(documents)
    doc_term_matrix = sparse_matrix.todense()
    df = pd.DataFrame(doc_term_matrix, columns=count_vectorizer.get_feature_names(), index=['x', 'y'])
    return cosine_similarity(df, df)[0,1]

result = beer[['beer', 'cleaned_review']]
result['cos_score'] = beer['cleaned_review'].map(cos)
result


# ### Task D. For every review, perform a sentiment analysis. 

# In[12]:


#! pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# In[13]:


analyzer = SentimentIntensityAnalyzer()

def sentimentScore(x):
    return analyzer.polarity_scores(x)['compound']

result['Sentiment_Score'] = result['cleaned_review'].astype('str').apply(sentimentScore)
result[:5]


# ### Task E. Recommend 3 products to the customer.
# 
# Assume an evaluation score for each beer = average similarity score + average sentiment score.
# 
# Now recommend 3 products to the customer. 

# In[14]:


#Assume an evaluation score for each beer = average similarity score + average sentiment score. 
recommend1 = result.groupby(['beer'])[['cos_score','Sentiment_Score']].mean()
recommend1['evaluation_score'] = recommend1['cos_score'] + recommend1['Sentiment_Score']
recommend1


# In[15]:


recommend1.sort_values(by='evaluation_score', ascending=False, inplace=True)
recommend1[:5]


# ### Task F. Use word vectors to recommend
# 
# How would our recommendation change if we use word vectors (the spaCy package would be the easiest to use with pretrained word vectors) instead of plain vanilla bag-of-words cosine similarity? One way to analyze the difference would be to consider the % of reviews that mention a preferred attribute. E.g., if we recommend a product, what % of its reviews mention an attribute specified by the customer? Any difference across bag-of-words and word vector approaches? This article may be useful: https://medium.com/swlh/word-embeddings-versus-bag-of-words-the-curious-case-of-recommender-systems-6ac1604d4424?source=friends_link&sk=d746da9f094d1222a35519387afc6338
# 
# Note that the article doesn’t claim that bag-of-words will always be better than word embeddings for recommender systems. It lays out conditions under which it is likely to be the case. That is, depending on the attributes we use, we may or may not see the same effect. 

# In[16]:


#!python -m spacy download en_core_web_md
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
nlp = spacy.load('en_core_web_md') 


# In[17]:


def spacy_cos(x):
    #spaCy uses word vectors for medium (md) and large (lg)
    text1 = x
    text2 = 'fruity complex balanced'
    
    #Calculates spaCy similarity between texts 1 and 2
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    return doc1.similarity(doc2)

result['spacy_cos_score'] = beer['cleaned_review'].map(spacy_cos)
result


# In[18]:


#Recommend 3 products to the customer.
#Assume an evaluation score for each beer = average similarity score + average sentiment score. 
recommend2 = result.groupby(['beer'])[['spacy_cos_score','Sentiment_Score']].mean()
recommend2['evaluation_score'] = recommend2['spacy_cos_score'] + recommend2['Sentiment_Score']
recommend2


# In[19]:


recommend2.sort_values(by='evaluation_score', ascending=False, inplace=True)
recommend2[:5]


# In[21]:


for i in ['Flora Plum', 'Genealogy Of Morals - Bourbon Barrel-Aged',
          "Hunahpu's Imperial Stout - Laird's Apple Brandy Barrel"]:
    a = 0
    df = beer[beer['beer']==i]['tokenized_review']
    for j in ['fruity', 'balanced', 'complex']:
        for k in range(len(df)):
            if j in df.iloc[k]:
                a = a+1
        print('Beer:',i, 'Attribute:',j, 'Perc_of_reviews:',a/len(df))
    print()


# In[22]:


recom_beer_1 = ['Dorothy (Wine Barrel Aged)', 'Saison Bernice']
recom_beer_2 = ['All That Is And All That Ever Will Be', 'JJJuliusss!']
attributes = ['fruity', 'complex', 'balanced']


# In[23]:


for i in recom_beer_1:
    a = 0
    df = beer[beer['beer']==i]['tokenized_review']
    for j in attributes:
        for k in range(len(df)):
            if j in df.iloc[k]:
                a = a+1
        print('Beer:',i, 'Attribute:',j, 'Perc_of_reviews:',a/len(df))
    print()


# In[24]:


for i in recom_beer_2:
    a = 0
    df = beer[beer['beer']==i]['tokenized_review']
    for j in attributes:
        for k in range(len(df)):
            if j in df.iloc[k]:
                a = a+1
        print('Beer:',i, 'Attribute:',j, 'Perc_of_reviews:',a/len(df))
    print()


# **Conclusions:** 
# 
# 
# Because bag-of-words and word vector approaches give us the same top three bears, we further look at top 5 and see difference in the fourth and fifth beer recommended. Bag-of-words recommends 'Dorothy (Wine Barrel Aged)' and 'Saison Bernice'. Word vector recommends 'All That Is And All That Ever Will Be' and 'JJJuliusss!'. 
# 
# Then we calculate percentage of reviews that mention a preferred attributes for such two beers under bag-of-words and word vector approaches. It shows that more reviews mention one or more attributes specified by the customer under bag-of-words approaches.
# 
# We can see bag-of-words approache gives us lower cosine similarity but higher review mention rate, while word vector approach gives us higher cosine similarity but lower review mention rate. This is because bag-of-words looks for an exact match of words. And we would feel comfortable recommending a product to customer when as many as reviews of such product mention a feature that a shopper considers important.

# ### Task G. Simply chose the 3 highest rated products.
# 
# How would our recommendations differ if we ignored the similarity and feature sentiment scores and simply chose the 3 highest rated products from the entire dataset? Would these products meet the requirements of the user looking for recommendations? Why or why not? Use the similarity and sentiment scores as well as overall ratings to think of this question.
# 
# Here is a sample web implementation of a recommender system based on the same principles (runningshoe4you.com). But in this assignment, we will not build this type of full automation.

# In[25]:


df = beer.groupby(['beer'])[['rating']].mean()

combine = recommend1[['cos_score', 'evaluation_score']]
combine['Sentiment_Score'] = recommend1[['Sentiment_Score']]
combine[['spacy_cos_score', 'spacy_evaluation_score']] = recommend2[['spacy_cos_score', 'evaluation_score']]
combine['overall_rating'] = df


# In[26]:


combine.sort_values(by='overall_rating', ascending=False)


# In[27]:


for i in ['Chemtrailmix', 'Blessed', 'SR-71']:
    a = 0
    df = beer[beer['beer']==i]['tokenized_review']
    for j in attributes:
        for k in range(len(df)):
            if j in df.iloc[k]:
                a = a+1
        print('Beer:',i, 'Attribute:',j, 'Perc_of_reviews:',a/len(df))
    print()


# In[28]:


combine.loc[['Chemtrailmix', 'Blessed', 'SR-71','Flora Plum', 'Genealogy Of Morals - Bourbon Barrel-Aged',
             "Hunahpu's Imperial Stout - Laird's Apple Brandy Barrel"]]


# **Conclusions:** 
# 
# If we use highest rating to recommend product, then the top 3 beers would be 'Chemtrailmix', 'Blessed', and 'SR-71'. However, reviews of 'Chemtrailmix'don't mention 'fruity' and 'complex', reviews of 'Blessed'don't mention 'fruity', and reviews of 'SR-71'almost have nothing related to 'fruity', 'complex', and 'balanced'. So we can conlude these three products don't meet the attributes requirements of customers.
# 
# Based on cosine similarity and semtiment analysis, we would recommend 'Flora Plum', 'Genealogy Of Morals - Bourbon Barrel-Aged', and "Hunahpu's Imperial Stout - Laird's Apple Brandy Barrel" to customers who think 'fruity', 'complex', and 'balanced' are important features. Although these three beers have relatively lower overall rating, users discuss more about whether these beers are fruity, complex and/or balanced. And based on sentiment analysis, we can see that these three beers also receive more praise from users, which means they are more palatable to the markets.
# 
# Therefore, we would choose making recommendation to a customer by cosine similarity and semtiment analysis.

# In[ ]:




