#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing necessary modules
import pandas as pd 
from dask import delayed
import dask.dataframe as dd
#nltk.download('stopwords')
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer


# In[2]:


#Importing the data
data=dd.read_csv("PDS-2 Project.csv",sep=r"\t",blocksize=50e6)  #Using r to treat \t as a raw string
data1=pd.read_csv("PDS-2 Project.csv",sep=r"\t") 


# In[3]:


data.head()


# In[3]:


#Renaming columns
data.columns=['ID','Title','Abstract','Primary category','Secondary category','Tertiary category']
data1.columns=['ID','Title','Abstract','Primary category','Secondary category','Tertiary category']


# In[5]:


data1


# In[6]:


#Looking at primary category 
data1['Primary category'].value_counts(normalize=True)*100


# In[4]:


data['Abstract']=data['Abstract'].astype('str')
data1['Abstract']=data1['Abstract'].astype('str')


# In[8]:


# import dask.dataframe as dd
# from sklearn.feature_extraction.text import HashingVectorizer

# vectorizer = HashingVectorizer()

# # create a dask dataframe
# ddf = data

# # initialize the vectorizer with the first batch of data
# first_batch = ddf.head(10000)['Abstract']
# matrix = vectorizer.transform(first_batch)

# # convert the matrix to a dask dataframe
# matrix_df = dd.from_array(matrix.toarray())

# # compute the TF-IDF matrix in batches
# for batch in ddf.to_delayed()[1:]:
#     batch_matrix = vectorizer.transform(batch['Abstract'])
#     batch_matrix_df = dd.from_array(batch_matrix.toarray())
#     matrix_df = dd.concat([matrix_df, batch_matrix_df], axis=0)

# # print the resulting matrix DataFrame
# print(matrix_df.compute())



# In[9]:


# import dask.dataframe as dd
# from sklearn.feature_extraction.text import TfidfVectorizer
# import numpy as np
# import scipy.sparse as sp

# # define the chunk size
# chunk_size = 1000

# # create a dask dataframe
# ddf = data

# # define the vectorizer
# vectorizer = TfidfVectorizer()

# # define a function to apply the vectorizer to each chunk
# def vectorize_chunk(chunk):
#     matrix = vectorizer.transform(chunk['Abstract'])
#     return matrix

# # initialize the sparse matrix with the first chunk
# first_chunk = ddf.head(chunk_size)
# matrix = vectorize_chunk(first_chunk)
# matrix = matrix.astype(np.float32)

# # iterate over the remaining chunks and add to the sparse matrix
# for i in range(1, len(ddf)//chunk_size + 1):
#     chunk = ddf.iloc[i*chunk_size:(i+1)*chunk_size]
#     matrix_chunk = vectorize_chunk(chunk)
#     matrix_chunk = matrix_chunk.astype(np.float32)
#     matrix = sp.vstack([matrix, matrix_chunk], format='csr')

# # convert the matrix into a dask dataframe
# matrix_df = dd.from_array(matrix)

# # print the resulting matrix DataFrame
# print(matrix_df.compute())


# In[10]:


# #Lemmatizing the text column --> Lemmatizing preserves actual words instead of just the stem so we go with that 
# from nltk.stem import WordNetLemmatizer
  
# lemmatizer = WordNetLemmatizer()

# data['Abstract'] = data['Abstract'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))


# In[11]:


#--------------------------------------------------------
#Avg no of words in every document 

data1['word_count'] = data1['Abstract'].str.count(' ') + 1

# calculate the average number of words per abstract
avg_words = data1['word_count'].mean()

# print the result
print(f"The average number of words in each abstract is: {avg_words}")


# In[12]:


# create a new column with the count of words in each abstract
data1['word_count'] = data1['Abstract'].str.count(' ') + 1

# calculate the median number of words per abstract
avg_words = data1['word_count'].median()

# print the result
print(f"The average number of words in each abstract is: {avg_words}")


# In[13]:


#Looking at the distribution of words 
import matplotlib.pyplot as plt

# split the text in the "Abstract" column into a list of words
data1["word_counts"] = data1['Abstract'].str.split().str.len()

# word_lists=list(word_lists)
# # get the number of words in each record

# word_counts = word_lists.apply(len)

# plot a histogram of the word counts
plt.hist(data1["word_counts"], bins=100)
plt.xlabel('Number of words')
plt.ylabel('Frequency')
plt.show()


# In[14]:


#-------------------------------------
#Now reducing each record to contain only 105 words and running the processes again 

#data1['Abstract_v2'] = data1['Abstract'].str.split().str.slice(0, 20).str.join(' ')


# In[15]:


#Taking same distribution from every class

data1["Primary category"].value_counts(normalize=True)


# In[8]:


out = data1.groupby(['Primary category']).sample(frac=0.25)
print(out['Primary category'].value_counts(normalize=True))
print(data1['Primary category'].value_counts(normalize=True))
len(out)


# In[9]:


import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# define a function to lemmatize a sentence
def lemmatize_sentence(sentence):
    word_list = nltk.word_tokenize(sentence.lower())
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    return lemmatized_output

# apply the lemmatize_sentence function to the "Abstract" column in the "out" dataframe
out["Abstract_v2"] = out["Abstract"].apply(lambda x: lemmatize_sentence(x))

#Lemmatizer has reduced the words to their base forms 


# In[7]:


out[["Abstract_v2","Abstract"]]  #There's some difference


# In[8]:


# Activate CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer


# create an instance of the CountVectorizer class
vectorizer = CountVectorizer()

# fit and transform the "Abstract" column in the "out" dataframe to create the document-term matrix
doc_matrix = vectorizer.fit_transform(out["Abstract_v2"])

# get the list of feature names (i.e., the terms in the vocabulary)
feature_names = vectorizer.get_feature_names_out()


dense_matrix = doc_matrix.toarray()

# create a new pandas dataframe with the dense matrix and the list of feature names
doc_matrix_df = pd.DataFrame(dense_matrix, columns=feature_names)

# display the first 10 rows of the document-term matrix dataframe
print(doc_matrix_df.head(10))


# In[20]:


#Obviously a lot of scientific words
#Now let's look at the top 25 words in a distribution

import matplotlib.pyplot as plt

# compute the frequency of each word across all documents
word_freq = doc_matrix.sum(axis=0)

# create a list of tuples with each word and its frequency
word_freq_tuples = [(word, freq) for word, freq in zip(feature_names, word_freq.tolist()[0])]

# sort the list of tuples by frequency in descending order
word_freq_tuples = sorted(word_freq_tuples, key=lambda x: x[1], reverse=True)

# select the top 25 words with the highest frequency
top_words = word_freq_tuples[:25]

# extract the word and frequency information into separate lists
words = [t[0] for t in top_words]
freqs = [t[1] for t in top_words]

# create a bar chart of the word frequencies
plt.figure(figsize=(10, 6))
plt.bar(words, freqs)
plt.xticks(rotation=90)
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.title('Top 25 Words in Abstracts')
plt.show()


# # Moving on to a bit of modelling

# In[9]:


import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay


# In[1]:


#Building the TF-IDF Matrix 
features = out['Abstract_v2']

vectorizer = TfidfVectorizer(max_features=2500, min_df=10, max_df=0.8)

processed_features = vectorizer.fit_transform(features).toarray()


# In[24]:


out["Primary category"].value_counts(normalize=True)*100


# In[12]:


#Let's combine classes with less than 3 % data record in them into an "Other" class
# Calculate the value counts of the 'Abstract' column
value_counts = out['Primary category'].value_counts(normalize=True)

# Find the values that have less than 4% of the data
values_to_combine = value_counts[value_counts < 0.03].index.tolist()

# Replace those values with 'Other'
out.loc[out['Primary category'].isin(values_to_combine), 'Primary category'] = 'Other'


# In[26]:


out["Primary category"].value_counts(normalize=True)*100


# In[14]:


#Splitting into train test split 

labels = out['Primary category']

X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)


# X_train.shape

# In[15]:


#Using a basic random forest classifier 

text_classifier = RandomForestClassifier(n_estimators=50, random_state=100)
text_classifier.fit(X_train, y_train)


# In[16]:


predictions = text_classifier.predict(X_test)


# In[17]:


cm = confusion_matrix(y_test, predictions, labels=text_classifier.classes_)
print(cm)


# In[18]:


disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=text_classifier.classes_)


# In[19]:


disp.plot()


# In[20]:


print(classification_report(y_test,predictions))


# In[ ]:


#Overall accuracy is decent at 65 % but the F1 score of a few classes is really low 

#USING IMBLEARN TO DEAL WITH IMBALANCED CLASSES


# In[ ]:





# In[ ]:


#-----------------------------------


# In[21]:


#Topic modelling

doc_matrix.shape


# In[22]:


#USING LDA FOR Topic modelling
LDA = LatentDirichletAllocation(n_components=5, random_state=35)
LDA.fit(doc_matrix)


# In[ ]:





# In[23]:


from sklearn.feature_extraction.text import CountVectorizer

# initialize the CountVectorizer object
vectorizer = CountVectorizer()

# fit the vectorizer on your input data
vectorizer.fit(out["Abstract_v2"])

# extract the feature names
feature_names = vectorizer.get_feature_names_out()

# now you can use the feature names to extract the top terms for each topic


# In[24]:


for i,topic in enumerate(LDA.components_):
    print(f'Top 10 words for topic #{i}:')
    print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])
    print('\n')


# In[ ]:





# # Nmd Ent Rcgntn

# In[25]:


from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


# In[26]:


out2=out.head(200)

out2['NN'] = ''
out2['JJ'] = ''
out2['VB'] = ''
out2['GEO'] = ''

def tweet_ner(chunker):
    treestruct = ne_chunk(pos_tag(word_tokenize(chunker)))
    entitynn = []
    entityjj = []
    entityg_air = []
    entityvb = []
    for y in str(treestruct).split('\n'):
        if 'GPE' in y or 'GSP' in y:
            entityg_air.append(y)
        elif '/VB' in y:
            entityvb.append(y)
        elif '/NN' in y:
            entitynn.append(y)
        elif '/JJ' in y:
            entityjj.append(y)
    stringnn = ''.join(entitynn)
    stringjj = ''.join(entityjj)
    stringvb = ''.join(entityvb)
    stringg = ''.join(entityg_air)
    return stringnn, stringjj, stringvb, stringg


# In[27]:


i = 0
for x in out2['Abstract_v2']:
    entitycontainer = tweet_ner(x)
    out2.at[i,'NN'] = entitycontainer[0]
    out2.at[i,'JJ'] = entitycontainer[1]
    out2.at[i,'VB'] = entitycontainer[2]
    out2.at[i,'GEO'] = entitycontainer[3]
    i += 1


# In[29]:


out2['NN'].unique().tolist()
out2['JJ'].unique().tolist()
out2['VB'].unique().tolist()
out2['GEO'].unique().tolist()


# In[30]:


nn=list(out2['NN'])
jj=list(out2['JJ'])
vb=list(out2['VB'])


# In[31]:


import matplotlib.pyplot as plt
from collections import Counter


# TOP 10 NOUNS

# In[32]:


word_list = []
for string in nn:
    string = string.strip()
    words = string.split(' ')
    word_list.extend(words)
    
word_list=[j for i,j in enumerate(word_list) if j!='']
word_list=[j for i,j in enumerate(word_list) if j!=',']
word_list=[j for i,j in enumerate(word_list) if j!=',']
word_list=[j.replace("/NN","") for i,j in enumerate(word_list)]
word_list=[j.replace(")","") for i,j in enumerate(word_list)]
word_list=[j.replace("(","") for i,j in enumerate(word_list)]


import matplotlib.pyplot as plt
from collections import Counter

# Create a Counter object to count the frequency of each word
word_counts = Counter(word for word in word_list if len(word) > 3)

# Get the top 10 frequent words
top_words = dict(word_counts.most_common(10))

# Plot the bar graph
plt.bar(top_words.keys(), top_words.values())
plt.title("Top 10 Frequent Words (More than 3 letters)")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()


# TOP 10 VERBS 

# In[33]:


word_list = []
for string in vb:
    string = string.strip()
    words = string.split(' ')
    word_list.extend(words)
    
word_list=[j for i,j in enumerate(word_list) if j!='']
word_list=[j for i,j in enumerate(word_list) if j!=',']
word_list=[j for i,j in enumerate(word_list) if j!=',']
word_list=[j.replace("/VB","") for i,j in enumerate(word_list)]
word_list=[j.replace("N","") for i,j in enumerate(word_list)]
word_list=[j.replace("G","") for i,j in enumerate(word_list)]
word_list=[j.replace(")","") for i,j in enumerate(word_list)]
word_list=[j.replace("(","") for i,j in enumerate(word_list)]


import matplotlib.pyplot as plt
from collections import Counter

# Create a Counter object to count the frequency of each word
word_counts = Counter(word for word in word_list if len(word) > 3)

# Get the top 10 frequent words
top_words = dict(word_counts.most_common(10))

# Plot the bar graph
plt.bar(top_words.keys(), top_words.values())
plt.title("Top 10 Frequent Words (More than 3 letters)")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()


# TOP 10 ADJECTIVES

# In[34]:


word_list = []
for string in jj:
    string = string.strip()
    words = string.split(' ')
    word_list.extend(words)
    
word_list=[j for i,j in enumerate(word_list) if j!='']
word_list=[j for i,j in enumerate(word_list) if j!=',']
word_list=[j for i,j in enumerate(word_list) if j!=',']
word_list=[j.replace("/JJ","") for i,j in enumerate(word_list)]
word_list=[j.replace("JJ","") for i,j in enumerate(word_list)]
word_list=[j.replace(")","") for i,j in enumerate(word_list)]
word_list=[j.replace("(","") for i,j in enumerate(word_list)]


import matplotlib.pyplot as plt
from collections import Counter

# Create a Counter object to count the frequency of each word
word_counts = Counter(word for word in word_list if len(word) > 3)

# Get the top 10 frequent words
top_words = dict(word_counts.most_common(10))

# Plot the bar graph
plt.bar(top_words.keys(), top_words.values())
plt.title("Top 10 Frequent Words (More than 3 letters)")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()


# # Use 2 pretrained models for sentiment analysis 
# 

# In[5]:


from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)


# In[37]:


# Step 1: Load the pre-trained DilBERT model and tokenizer

out2=out.groupby(['Primary category']).sample(frac=0.10)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "monologg/koelectra-base-v3-discriminator"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Step 2: Load the dataset and extract the "Abstract_v2" column
import pandas as pd


text_data = out2["Abstract_v2"].tolist()

# Step 3: Tokenize the text data
tokenized_data = tokenizer(text_data, padding=True, truncation=True, return_tensors="pt")

# Step 4: Encode the tokenized data as input features
input_ids = tokenized_data["input_ids"]
attention_mask = tokenized_data["attention_mask"]

# Step 5: Use the pre-trained DilBERT model to predict the sentiment labels
outputs = model(input_ids, attention_mask=attention_mask)
predicted_labels = outputs.logits.argmax(dim=1)

# Step 6: Evaluate the model's performance
true_labels = df["Label"].tolist()
accuracy = (predicted_labels == true_labels).float().mean()
print(f"Accuracy: {accuracy:.4f}")


# In[ ]:


#Using tidytext
get_ipython().system('pip install tidytext')


# In[ ]:


nltk.download("punkt")


# In[10]:


out2 = out.groupby(['Primary category']).sample(frac=0.25)

#Positive-Negative
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#calculate the negative, positive, neutral and compound scores, plus verbal evaluation
def sentiment_vader(sentence):

    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(sentence)
    negative = sentiment_dict['neg']
    positive = sentiment_dict['pos']

    if sentiment_dict['compound'] >= 0.05 :
        overall_sentiment = "Positive"
    else :
        overall_sentiment = "Negative"
  
    return negative, positive,overall_sentiment


out2['Sentiment'] = out2['Abstract_v2'].apply(sentiment_vader)


# In[12]:


out2


# In[13]:


#Strong-weak
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def sentiment_vader(sentence, threshold=0.05):
    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(sentence)
    negative = sentiment_dict['neg']
    neutral = sentiment_dict['neu']
    positive = sentiment_dict['pos']
    compound = sentiment_dict['compound']

    if abs(sentiment_dict['compound']) >= threshold:
        overall_sentiment = "Strong"
    else:
        overall_sentiment = "Weak"
  
    return negative, positive, compound, overall_sentiment


out2['Sentiment2'] = out2['Abstract_v2'].apply(sentiment_vader)


# In[14]:


#Argumentative vs descriptive 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def classify_emotion(sentence):
    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(sentence)
    negative = sentiment_dict['neg']
    neutral = sentiment_dict['neu']
    positive = sentiment_dict['pos']
    compound = sentiment_dict['compound']

    if compound > 0.1:
        emotion = "Argumentative"
    else:
        emotion = "Descriptive"
  
    return negative, positive, compound, emotion

out2["Sentiment3"]=out2["Abstract_v2"].apply(classify_emotion)


# In[15]:


#Technical - Non technical 

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def classify_technicality(sentence):
    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(sentence)
    negative = sentiment_dict['neg']
    neutral = sentiment_dict['neu']
    positive = sentiment_dict['pos']
    compound = sentiment_dict['compound']

    if compound >= 0.1:
        technicality = "Technical"
    else:
        technicality = "Non-technical"
  
    return negative, positive, compound, technicality


out2["Sentiment4"]=out2["Abstract_v2"].apply(classify_technicality)


# In[16]:


out2


# In[77]:


#Wordcloud 
# Import the necessary libraries
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

pos=out2[out2["Sentiment"]=='  Positive']

# Extract the text column
text = ' '.join(pos['Abstract_v2'].astype('str'))

# Generate the wordcloud
wordcloud = WordCloud(width = 800, height = 800, background_color ='white', min_font_size = 10).generate(text)

# Plot the wordcloud
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)

# Display the plot
plt.show()


# In[76]:


out2[out2["Sentiment"]=="  Positive"]


# In[69]:


out2['Sentiment'].value_counts()


# In[38]:


import re

out2["Sentiment"]=out2["Sentiment"].apply(lambda x : re.sub('[^a-zA-Z\s]', '', str(x)))

out2["Sentiment2"]=out2["Sentiment2"].apply(lambda x : re.sub('[^a-zA-Z\s]', '', str(x)))
out2["Sentiment3"]=out2["Sentiment3"].apply(lambda x : re.sub('[^a-zA-Z\s]', '', str(x)))
out2["Sentiment4"]=out2["Sentiment4"].apply(lambda x : re.sub('[^a-zA-Z\s]', '', str(x)))


# In[34]:


(out2["Sentiment"].value_counts(normalize=True)*100).plot(kind="bar")


# In[44]:


(out2["Sentiment2"].value_counts(normalize=True)*100).plot(kind="bar",color='green')


# In[45]:


(out2["Sentiment3"].value_counts(normalize=True)*100).plot(kind="bar",color='orange')


# In[46]:


(out2["Sentiment4"].value_counts(normalize=True)*100).plot(kind="bar",color='red')


# In[61]:


out2[['Abstract_v2','Sentiment','Sentiment2','Sentiment3','Sentiment4']].tail(2)


# In[60]:


out2[['Abstract_v2']].head(1)


# In[ ]:




