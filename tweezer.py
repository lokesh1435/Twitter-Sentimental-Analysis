
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 


df1 = pd.read_csv('train (1).csv')

df1.head()

df1.isna().sum()

print(len(df1[df1.label == 0]), 'Non-Hatred Tweets')
print(len(df1[df1.label == 1]), 'Hatred Tweets')


import nltk
from sklearn import re  
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
lemma = WordNetLemmatizer()
from wordcloud import WordCloud, STOPWORDS
from nltk import FreqDist 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix, classification_report, f1_score 

def normalizer(tweet):
	tweets = " ".join(filter(lambda x: x[0]!= '@' , tweet.split()))
	tweets = re.sub('[^a-zA-Z]', ' ', tweets)
	tweets = tweets.lower()
	tweets = tweets.split()
	tweets = [word for word in tweets if not word in set(stopwords.words('english'))]
	tweets = [lemma.lemmatize(word) for word in tweets]
	tweets = " ".join(tweets)
	return tweets

df1['normalized_text'] = df1.tweet.apply(normalizer)

def extract_hashtag(tweet):
    tweets = " ".join(filter(lambda x: x[0]== '#', tweet.split()))
    tweets = re.sub('[^a-zA-Z]',' ',  tweets)
    tweets = tweets.lower()
    tweets = [lemma.lemmatize(word) for word in tweets]
    tweets = "".join(tweets)
    return tweets


df1['hashtag'] = df1.tweet.apply(extract_hashtag)


df1.head()

# all tweets 
all_words = " ".join(df1.normalized_text)
#print(all_all_words)
#Hatred tweets
hatred_words = " ".join(df1[df1['label']==1].normalized_text)
#print(hatred_words)

wordcloud = WordCloud(height=2000, width=2000, stopwords=STOPWORDS, background_color='white')
wordcloud = wordcloud.generate(all_words)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

wordcloud = WordCloud(height=2000, width=2000, stopwords=STOPWORDS, background_color='white')
wordcloud = wordcloud.generate(hatred_words)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


freq_all_hashtag = FreqDist(list((" ".join(df1.hashtag)).split())).most_common(10)

freq_hatred_hashtag = FreqDist(list((" ".join(df1[df1['label']==1]['hashtag'])).split())).most_common(10)


df_allhashtag = pd.DataFrame(freq_all_hashtag, columns=['words', 'frequency'])
df_hatredhashtag = pd.DataFrame(freq_hatred_hashtag, columns=['words', 'frequency'])
print(df_allhashtag.head())


sns.barplot(x='words', y='frequency', data=df_allhashtag)
plt.xticks(rotation = 45)
plt.title('hashtag words frequency')
plt.show()

sns.barplot(x='words', y='frequency', data=df_hatredhashtag)
plt.xticks(rotation = 45)
plt.title('hatred hashtag frequency')
plt.show()


len(df1)
corpus = []
for i in range(0,1000):
    corpus.append(df1['normalized_text'][i])
cv = CountVectorizer(stop_words=stopwords.words('english'))
cv.fit(corpus)


X = cv.transform(corpus).toarray()
y = df1.iloc[:,1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

classifier1 = LogisticRegression(C=10)
classifier1.fit(X_train, y_train)

y_pred = classifier1.predict(X_test)
y_prob = classifier1.predict_proba(X_test)
print(f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


tfidf = TfidfVectorizer(ngram_range=(1,3), min_df=10, stop_words=stopwords.words('english'))
X1 = tfidf.fit_transform(corpus)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, test_size=0.33, random_state=42)
classifier2 = LogisticRegression(C=10)
classifier2.fit(X1_train, y1_train)


y1_pred = classifier2.predict(X1_test)
y1_prob = classifier2.predict_proba(X1_test)
print(f1_score(y1_test, y1_pred))
print(classification_report(y1_test, y1_pred))
print(confusion_matrix(y1_test, y1_pred))


threshold = np.arange(0.1,0.9,0.1)
score = [f1_score(y1_test, ((y1_prob[:,1] >= x).astype(int))) for x in threshold]
plt.plot(threshold, score)
plt.xlabel('Threshold Probability')
plt.ylabel('F1 score')
plt.show()


df2 = pd.read_csv('test(1-200).csv')
df2.head()


df2['normalized_text'] = df2['tweet'].apply(normalizer)
# creating corpus
corpus_test = []
for i in range(0,200):
    corpus_test.append(df2.normalized_text[i])
#corpus_test

Test_X = tfidf.transform(corpus_test)
pred_Y = classifier2.predict(Test_X)
prob_Y = classifier2.predict_proba(Test_X)
df2['pred_label'] = pred_Y
scores = (prob_Y[:,1] >= 0.5).astype(int)
df2['score'] = scores

print(df2[df2.pred_label == 1])	
print(df2[df2.pred_label == 0])				


