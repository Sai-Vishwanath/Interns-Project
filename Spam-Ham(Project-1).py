#!/usr/bin/env python
# coding: utf-8

# ## <span style='color:#00337C'>Project Name - Spam/Ham Detection</span>
# ### <span style='color:#00337C'>Done By - B.Sai Vishwanath</span>
# ### <span style='color:#00337C'>ID-ACG2410101304122</span>

# ## Loading Libraries 

# ### Importing NLTK,Pandas,Numpy,Seasborn,Matplot,Scikit Learn 

# In[2]:


import nltk 
import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords

# For Visual representation purposes we import matplot and seasborn,Word Cloud and etc

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
get_ipython().run_line_magic('matplotlib', 'inline')

from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split 
from sklearn import metrics


# ## Importing The Dataset
# 

# In[3]:


dataset = pd.read_csv("C:/Users/vissu/Downloads/Spam-Ham Dataset.csv", encoding='latin1')
dataset.head(10)


# In[4]:


dataset.tail()


# ## Remove the unnecessary columns for dataset and rename the column names.

# In[5]:


dataset=dataset.drop(labels = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1)
dataset.columns = ["label", "message"]


# In[6]:


dataset.head()


# # **Data Analysis**
# 
# 
# **<span style='color:#0079FF'>Let's check out some of the stats with some plots and the built-in methods in pandas!</span>**

# In[7]:


dataset.info()


# **<span style='color:#059212'>There are total 5572 SMS in this dataset with 2 columns label and message.</span>**

# In[8]:


dataset.describe()


# *  **<span style='color:#059212'>There is two unique labels.</span>**
# *  **<span style='color:#059212'>There are some repeated messages as unique is less that the count due to some comman messages.</span>**

# ## <span style='color:#0079FF'>Let's use groupby to use describe by label, this way we can begin to think about the features that separate ham and spam!</span>
# 
#  

# In[9]:


dataset.groupby('label').describe().T


# * <span style='color:#059212'>4825 ham messages out of which 4516 are unique..</span>
# * <span style='color:#059212'>747 span messages out of which 653 are unique.</span>
# * <span style='color:#059212'>"Sorry, I'll call later" is the most popular ham message with repetition of 30 times.</span>
# * <span style='color:#059212'>Please call our customer service representativ..." is the most popular spam message with repetition 4 times.</span>

# -  <span style='color:#0079FF'>As we continue our analysis we want to start thinking about the features we are going to be using. This goes along with the general idea of feature engineering. The better the domain knowledge, better the ability to engineer more features from it.</span>
# 
#   --->Let's make a new feature to detect how long the text messages are:

# In[10]:


dataset['length'] = dataset['message'].apply(len)
dataset.head()


# In[11]:


# Count the frequency of top 5 messages.

dataset['message'].value_counts().rename_axis(['message']).reset_index(name='counts').head()


# # <span style='color:#00337C'>Data Visualization</span>
# <span style='color:#0079FF'>Let's visualize this!</span>

# In[12]:


colors=['#28DF99','#0D7377']
dataset["label"].value_counts().plot(kind = 'pie',explode=[0, 0.1],figsize=(6, 6),autopct='%1.1f%%',shadow=True,colors=colors)
plt.title("Spam vs Ham")
plt.legend(["Ham", "Spam"])
plt.show()


# * <span style='color:#B3005E'>From The above Visual represtation of our data We find out that A lot of messages are actually not spam. About 87% of our dataset consists of normal messages.</span>

# In[13]:


plt.figure(figsize=(12,6))
dataset['length'].plot(bins=100, kind='hist',color='#000000') # with 100 length bins (100 length intervals) 
plt.title("Frequency Distribution of Message Length")
plt.xlabel("Length")
plt.ylabel("Frequency")


# In[14]:


dataset['length'].describe()


# **<span style='color:#059212'>From The Above analysis regarding the length of the messages we found that there a message with almost 910 characters in it.So Lets Check Out The Message.</span>**

# In[15]:


dataset[dataset['length'] == 910]['message'].iloc[0]


# _

# **<span style='color:#059212'>But let's focus back on the idea of trying to see if message length is a distinguishing feature between ham and spam:</span>**

# In[16]:


dataset.hist(column='length', by='label', bins=50,figsize=(12,4),color='#000000')


# **<span style='color:#059212'>Looks like spam messages are usually longer. Maybe message length can become a feature to predict whether the message is spam/ ham ?</span>**

# # <span style='color:#00337C'>Text Pre-processing</span>

# In[17]:


def text_preprocess(mess): <span style='color:#00337C'>
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Checking  characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]
    

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    nopunc = nopunc.lower()
    
    # Now just remove any stopwords and non alphabets
    nostop=[word for word in nopunc.split() if word.lower() not in stopwords.words('english') and word.isalpha()]
    
    return nostop


# Now let's "tokenize" these spam or ham messages. Tokenization is just the term used to describe the process of converting the normal text strings in to a list of tokens (words that we actually want).
# 
# Let's see an example output on on column:

# In[18]:


spam_messages = dataset[dataset["label"] == "spam"]["message"]
ham_messages = dataset[dataset["label"] == "ham"]["message"]
print("No of spam messages : ",len(spam_messages))
print("No of ham messages : ",len(ham_messages))


# **<span style ='color:#0079FF'>Now lets see, what are the major words that are detetcted as spam in the messages in a Wordcloud Manner**</span> 

# In[24]:


spam_words = text_preprocess(spam_messages)


# In[115]:


spam_wordcloud = WordCloud(width=600, height=400,colormap='cool',background_color='#ffffff').generate(' '.join(spam_words))
plt.figure( figsize=(10,8), facecolor='black')
plt.imshow(spam_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# **<span style ='color:#0079FF'>Wordcloud for spam messages shows that words like call, txt, win, free, reply, mobile, text etc. are widely used, let's check them statistically.</span>**

# In[33]:


print("Top 10 Spam words are :\n")
print(pd.Series(spam_words).value_counts().head(10))


# ### Wordcloud for Ham Messages

# In[34]:


ham_words = text_preprocess(ham_messages)


# In[118]:


ham_wordcloud = WordCloud(width=600, height=400,colormap='cool',background_color='#ffffff').generate(' '.join(ham_words))
plt.figure( figsize=(10,8), facecolor='k')
plt.imshow(ham_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# **<span style='color:#0079FF'>Wordcloud for ham messages shows that words like got, come, go, ur, know, call etc. are widely used, let's check them statistically.</span>**

# In[37]:


print("Top 10 Ham words are :\n")
print(pd.Series(ham_words).value_counts().head(10))


# #  <span style='color:#00337C'>Data Transformation</span>

# In[39]:


dataset.head()


# In[40]:


# Lets remove punctuations/ stopwords from all SMS 
dataset["message"] = dataset["message"].apply(text_preprocess)


# In[41]:


# Conver the SMS into string from list
dataset["message"] = dataset["message"].agg(lambda x: ' '.join(map(str, x)))


# In[42]:


dataset.head()


# #  <span style='color:#00337C'>Normalization</span> 
# **<span style='color:#0079FF'>There are a lot of ways to continue normalizing the text. Such as Stemming or Lemmatization.Our Goal is to bring a word to its root form.</span>**

# ## <span style='color:#00337C'>Vectorization</span> 
# 
# **<span style='color:#0079FF'>Currently, we have the messages as lists of tokens (also known as lemmas) and now we need to convert each of these messages into a vector the SciKit Learn's algorithm models can work with.</span>**
# 
# **<span style='color:#0079FF'>Now we'll convert each message, represented as a list of tokens (lemmas) above, into a vector that machine learning models can understand.</span>**

# In[46]:


# Creating the Bag of Words

# Note the here we are passing already process messages (after removing punctuations and stopwords)

vectorizer = CountVectorizer()
bow_transformer = vectorizer.fit(dataset['message'])

print("20 Bag of Words (BOW) Features: \n")
print(vectorizer.get_feature_names_out()[20:40])

print("\nTotal number of vocab words : ",len(vectorizer.vocabulary_))


# **<span style='color:#0079FF'>Let's take one text message and get its bag-of-words counts as a vector, putting to use our new bow_transformer:**</span>

# In[47]:


message4 = dataset['message'][3]
print(message4)


# **<span style='color:#0079FF'>Now let's see its vector representation:</span>**

# In[48]:


# fit_transform : Learn the vocabulary dictionary and return term-document matrix.
bow4 = bow_transformer.transform([message4])
print(bow4)
print(bow4.shape)


# **<span style='color:#0079FF'>This means that there are seven unique words in message number 4 (after removing common stop words). Let's go ahead and check and confirm which ones appear twice:</span>**

# In[50]:


print(bow_transformer.get_feature_names_out()[5945])


# **<span style='color:#0079FF'>Now we can use .transform on our Bag-of-Words (bow) transformed object and transform the entire DataFrame of messages. Let's go ahead and check out how the bag-of-words counts for the entire SMS corpus is a large, sparse matrix:</span>**

# In[52]:


messages_bow = bow_transformer.transform(dataset['message'])


# In[53]:


print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)


# ### TF_IDF
# TF-IDF stands for term frequency-inverse document frequency, and the tf-idf weight is a weight often used in information retrieval and text mining. This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus.

# In[54]:


from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)


# **<span style='color:#0079FF'>Let's try classifying our single random message and checking how we do:</span>**

# In[55]:


tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)


# In[57]:


print(bow_transformer.get_feature_names_out()[5945])
print(bow_transformer.get_feature_names_out()[3141])


# **<span style='color:#0079FF'>We'll go ahead and check what is the IDF (inverse document frequency) of the "say"?</span>**

# In[58]:


print(tfidf_transformer.idf_[bow_transformer.vocabulary_['say']])


# ## To transform the entire bag-of-words corpus into TF-IDF corpus at once:

# In[59]:


messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)


# In[61]:


dataset["message"][:10]


# **<span style='color:#0079FF'>Lets convert our clean text into a representation that a machine learning model can understand. I'll use the Tfifd for this.</span>**

# In[63]:


from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(encoding = "latin-1", strip_accents = "unicode", stop_words = "english")
features = vec.fit_transform(dataset["message"])
print(features.shape)

print(len(vec.vocabulary_))


# # <span style='color:#00337C'>Model Evaluation</span>
# 
# **<span style='color:#0079FF'>With messages represented as vectors, we can finally train our spam/ham classifier. Now we can actually use almost any sort of classification algorithms. For a variety of reasons, the Naive Bayes classifier algorithm is a good choice.</span>**
# 
# **<span style='color:#0079FF'>We'll be using scikit-learn here, choosing the Naive Bayes classifier to start with:</span>**

# ## Train Test Split

# In[67]:


msg_train, msg_test, label_train, label_test = \
train_test_split(messages_tfidf, dataset['label'], test_size=0.3)


# In[68]:


print("train dataset features size : ",msg_train.shape)
print("train dataset label size", label_train.shape)

print("\n")

print("test dataset features size", msg_test.shape)
print("test dataset lable size", label_test.shape)


# **<span style='color:#059212'>The test size is 30% of the entire dataset (1672 messages out of total 5572), and the training is the rest (3900 out of 5572).</span>**

# ## **<span style='color:#00337C'>Building Naive Bayes classifier Model</span>**
# **<span style='color:#0079FF'>Let's create a Naive Bayes classifier Model using Scikit-learn.**</span>

# In[69]:


from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
spam_detect_model = clf.fit(msg_train, label_train)


# In[70]:


predict_train = spam_detect_model.predict(msg_train)


# In[71]:


print("Classification Report \n",metrics.classification_report(label_train, predict_train))
print("\n")
print("Confusion Matrix \n",metrics.confusion_matrix(label_train, predict_train))
print("\n")
print("Accuracy of Train dataset : {0:0.3f}".format(metrics.accuracy_score(label_train, predict_train)))


# ### **<span style='color:#0079FF'>Let's try classifying our single random message and checking how we do:</span>**

# In[73]:


print('predicted:', spam_detect_model.predict(tfidf4)[0])
print('expected:', dataset['label'][3])


# # <span style='color:#00337C'>Model Evaluation</span>
# **<span style='color:#0079FF'>Now we want to determine how well our model will do overall on the entire dataset. Let's begin by getting all the predictions:</span>**

# In[74]:


label_predictions = spam_detect_model.predict(msg_test)
print(label_predictions)


# **<span style='color:#0079FF'>We can use SciKit Learn's built-in classification report, which returns precision, recall f1-score, and a column for support (meaning how many cases supported that classification).**</span>

# In[75]:


print(metrics.classification_report(label_test, label_predictions))
print(metrics.confusion_matrix(label_test, label_predictions))


# **There are quite a few possible metrics for evaluating model performance. Which one is the most important depends on the task and the business effects of decisions based off of the model. For example, the cost of mis-predicting "spam" as "ham" is probably much lower than mis-predicting "ham" as "spam".**

# In[76]:


# Printing the Overall Accuracy of the model
print("Accuracy of the model : {0:0.3f}".format(metrics.accuracy_score(label_test, label_predictions)))


# ### **<span style='color:#059212'>As The Accuracy is greater than 95% It passess all industry standard requirements,So Our Model Is Performing Well On The Dataset</span>**
