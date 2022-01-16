# Unstructured Data Analytics Project
Text Analysis_Beer Recommendation


# Introduction
The objective of this project is to create the building blocks of a crowdsourced recommendation system. This recommendation system should accept user inputs about desired attributes of a product and come up with 3 recommendations.

Obtain reviews of craft beer from beeradvocate.com. We would suggest using the following link, which shows the top 250 beers sorted by ratings: 
https://www.beeradvocate.com/beer/top-rated/

The nice feature of the above link is that it is a single-page listing of 250 top-rated beers (avoids the pagination feature, which you need in cases where listings go on for many pages). The way beeradvocate.com organizes reviews is that it provides about 25 reviews per page. The output file should have 3 columns: product_name, product_review, and user_rating.

Next, we perform analysis according to the following steps:
- Extract about 5-6k reviews
- Specify 3 attributes in a product
- Perform a similarity analysis using cosine similarity with the 3 attributes
- For every review, perform a sentiment analysis
- Recommend 3 products to the customer
- Use word vectors technique to recommend
- Compare bag-of-words and word vector approaches
- Simply chose the 3 highest rated products while ignoring the similarity and feature sentiment scores
- Analyze whether these products meet the requirements of the user looking for recommendations
