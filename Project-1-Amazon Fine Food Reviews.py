
# coding: utf-8

# # Project on classifying whether a review is positive or not for Amazon Fine Foods

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

#insert required modules
import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve,auc
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

import nltk
nltk.download('stopwords')


# In[2]:


#load the database file
con=sqlite3.connect('D:\Applied AI Course\database.sqlite')


# In[3]:


#query files
filt_data=pd.read_sql_query("""SELECT * FROM REVIEWS WHERE score!=3""",con)


# In[4]:


#check data and shape
print(filt_data.shape)
print(filt_data.head())


# In[5]:


import pickle

def savetofile(obj,filename):
    pickle.dump(obj,open(filename+".p","wb"), protocol=4)
def openfromfile(filename):
    temp = pickle.load(open(filename+".p","rb"))
    return temp


# In[6]:


#change the Score field to Review and assign as positive or negative either using lambdas or a custom function
#using custom function
def partition(x):
    if x < 3:
        return 'Negative'
    return 'Positive'

#using lambdas
#filt_data['Score']=filt_data['Score'].apply(lambda x: 'Positive' if int(x)>3 else 'Negative')


# In[7]:


#change column
ActScore=filt_data['Score']
positiveNegative=ActScore.map(partition)
filt_data['Score']=positiveNegative


# In[146]:


filt_data.head(3)


# In[9]:


filt_data.shape


# In[10]:


import gensim
from gensim.models import word2vec,KeyedVectors


# # Data cleaning-removing duplicate entries

# In[11]:


dup_data=pd.read_sql_query("""SELECT * FROM REVIEWS WHERE score!=3 ORDER BY ProductId """,con)


# In[12]:


dup_data.head(3)


# In[145]:


#the product id 0006641040 is a book and not a fine food and hence to be removed
sort_data=filt_data.sort_values('ProductId',axis=0,ascending=True)
sort_data.head(5)


# In[14]:


final_data=sort_data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"},keep='first',inplace=False)


# In[15]:


percent=(final_data['Id'].size*1.0 / filt_data['Id'].size*1.0) *100
print(percent)


# In[16]:


final_data["Score"].value_counts()


# In[143]:


final_data.head(3)


# In[18]:


dup_data1=pd.read_sql_query("""SELECT DISTINCT ProductId,UserId FROM REVIEWS WHERE score!=3 AND ProductId='0006641040' ORDER By ProductId """,con)


# In[19]:


dup_data1.shape


# In[22]:


labels=final_data['Score']


# In[24]:


labels.head(3)


# # BOW,TFIDF,Word2Vec(Avg-W2Vec,TfIDF-W2Vec) t-SNE plots
Text preprocessing
1) Remove HTML tags present in Text column words
2) remove any punctuation
3) check if word in english and alphanumeric
4) check if length>2
5) convert all words to lowercase
6) remove stopwords

# In[25]:


#helper functions

stop_word=set(stopwords.words('english'))
sno=SnowballStemmer('english')

def cleanhtml(sentence):
    cleanh=re.compile('<.*?>')
    cleantext=re.sub(cleanh,' ',sentence)
    return cleantext

def cleanpunc(sentence):
    cleaned=re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned=re.sub(r'[.|,|)|(|\|/]',r'',cleaned)
    return cleaned

print(stop_word)
print("##################################################")
print(sno.stem('tasty'))


# In[26]:


#code to check for implemented checks above
i=0
str1=''
final_string=[]
all_pos_words=[]
all_neg_words=[]
s=''

for sent in final_data['Text'].values:
    filtered_sentences=[]
    sent=cleanhtml(sent)
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):
                if(cleaned_words.lower() not in stop_word):
                    s=(sno.stem(cleaned_words.lower())).encode('utf8')
                    filtered_sentences.append(s)
                    if (labels.values)[i]=='Positive':
                        all_pos_words.append(s)
                    if (labels.values)[i]=='Negative':
                        all_neg_words.append(s)
                else:
                    continue
            else:
                continue
        
    str1=b" ".join(filtered_sentences)
       
    final_string.append(str1)
    i +=1
        
        
                    
                    


# In[27]:


final_data['CleanedText']=final_string #adding a column of CleanedText which displays the data after pre-processing of the review 
final_data['CleanedText']=final_data['CleanedText'].str.decode("utf8")


# In[142]:


final_data.head(3)


# In[29]:


#save it to database
conn=sqlite3.connect('final2.sqlite')
c=conn.cursor()
#c.execute("alter table REVIEWS add column '%s'" %labels)
conn.commit()
conn.text_factory=str
final_data.to_sql('Reviews',conn,schema=None,if_exists='replace')


# In[30]:


final_data.head(3)


# In[119]:


n_samples=2000
test_data=final_data.sample(n_samples)
label_data=final_data['Score'][0:2000]


# In[141]:


test_data.head(5)


# In[120]:


#bag of words
count_vect=CountVectorizer()
final_count=count_vect.fit_transform(test_data['CleanedText'].values)
type(final_count)
final_count.get_shape()
#Bi-grams and n-grams

freq_dist_pos=nltk.FreqDist(all_pos_words)
freq_dist_neg=nltk.FreqDist(all_neg_words)
print("Most common positive words:",freq_dist_pos.most_common(20))
print("Most common negative words:",freq_dist_neg.most_common(20))

#Bi-grams
#count_vect=CountVectorizer(ngram_range=(1,2))
#final_count=count_vect.fit_transform(test_data['CleanedText'].values)


# # Bag Of Words

# In[32]:


#bag of words
count_vect=CountVectorizer()
final_count=count_vect.fit_transform(final_data['CleanedText'].values)
print("the type of count vectorizer is:",type(final_count))
final_count.get_shape()


# In[33]:


final_count.get_shape


# In[34]:


#t-SNE plot for Bag of words
#from sklearn.preprocessing import StandardScaler

#standard_data=StandardScaler(with_mean=False).fit_transform(final_count)
#standard_data.shape


# In[35]:


n_samples=1000
std_data=final_count[0:n_samples,:n_samples].todense()
label_data=final_data["Score"][0:n_samples]


# In[36]:


std_data.shape


# In[37]:


from sklearn.manifold import TSNE

tmodel=TSNE(n_components=2,random_state=0,perplexity=30,n_iter=1000)
tsne_data=tmodel.fit_transform(std_data)

tsne_data = np.vstack((tsne_data.T, label_data)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("dim1", "dim2", "score"))


sns.FacetGrid(tsne_df, hue="score", size=6).map(plt.scatter, 'dim1', 'dim2').add_legend()
plt.title("TSNE for Bag Of Words")
plt.show()


# # TF-IDF

# In[38]:


#tf_idf_vect=TfidfVectorizer(ngram_range=(1,2))
tf_idf_vect=TfidfVectorizer()
final_tf_idf_vect=tf_idf_vect.fit_transform(final_data["CleanedText"].values)
final_tf_idf_vect.get_shape()
#get features
features=tf_idf_vect.get_feature_names()
print(len(features))
print("type of count vectorizer :",type(final_tf_idf_vect))


# In[39]:


#top tdf-idf features code taken from https://buhrmann.github.io/tfidf-analysis.html
def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

top_tfidf = top_tfidf_feats(final_tf_idf_vect[1,:].toarray()[0],features,25)


# In[40]:


top_tfidf


# In[41]:


#t-SNE visualization for tf-idf
n_samples=1000
std_data=final_tf_idf_vect[0:n_samples,:].todense()
label_data=final_data["Score"][0:n_samples]

#from sklearn.manifold import TSNE

tmodel=TSNE(n_components=2,random_state=0,perplexity=40,n_iter=1000)
tsne_data=tmodel.fit_transform(std_data)

tsne_data = np.vstack((tsne_data.T, label_data)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("dim1", "dim2", "score"))


sns.FacetGrid(tsne_df, hue="score", size=6).map(plt.scatter, 'dim1', 'dim2').add_legend()
plt.title("TSNE for TF-IDF")
plt.show()


# In[121]:


savetofile(final_tf_idf_vect,"tfidf")


# # Word2Vec

# In[65]:


pwd


# In[123]:


##Create own word2vec model

i=0
list_of_sentence=[]
for sent in test_data['CleanedText'].values:
    list_of_sentence.append(sent.split())
    #sent=cleanhtml(sent)
    #for w in sent.split():
        #for cleaned in cleanpunc(w).split():
            #if(cleaned.isalpha()):
                #filtered_sentence.append(cleaned.lower())
            #else:
                #continue
        #list_of_sentence.append(filtered_sentence)
print(test_data['CleanedText'].values[0])
print('###########')
print(list_of_sentence[0])
w2v_model=gensim.models.Word2Vec(list_of_sentence,min_count=5,size=50,workers=4)

words=list(w2v_model.wv.vocab)
print(len(words))


# In[128]:


w2v_model.save('w2vmodel')


# In[129]:


print(w2v_model)


# In[130]:


w2v_model.wv.most_similar('tasti')


# # Avg W2V

# In[132]:


#average word2vec
sent_vectors = [] 
for sent in list_of_sentence: # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length
    cnt_words =0 # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        if word in words:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
    if cnt_words != 0:
        sent_vec /= cnt_words
    sent_vectors.append(sent_vec)
print(len(sent_vectors))
print(len(sent_vectors[0]))

vec_avg=np.array(sent_vectors)


# In[134]:


#n_samples=1000
std_data=vec_avg
#label_data=final_data["Score"][0:n_samples]

#from sklearn.manifold import TSNE

tmodel=TSNE(n_components=2,random_state=0,perplexity=30,n_iter=1000)
tsne_data=tmodel.fit_transform(std_data)

tsne_data = np.vstack((tsne_data.T, label_data)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("dim1", "dim2", "score"))


sns.FacetGrid(tsne_df, hue="score", size=6).map(plt.scatter, 'dim1', 'dim2').add_legend()
plt.title("TSNE for Avg Word2Vec")
plt.show()

Maybe need to increase the sample size to get more correct design or dimension for the t-SNE
# # TF-IDF Word2Vec t-SNE

# In[135]:


tf_idf_vect=TfidfVectorizer(ngram_range=(1,2))
final_tf_idf_vect=tf_idf_vect.fit_transform(test_data["CleanedText"].values)
final_tf_idf_vect.get_shape()
#get features
features=tf_idf_vect.get_feature_names()
print(len(features))
print("type of count vectorizer :",type(final_tf_idf_vect))


# In[95]:


tf_idf=openfromfile('tfidf')
tf_idf


# In[136]:


print("shape:",final_tf_idf_vect.get_shape())


# In[137]:


from sklearn.decomposition import TruncatedSVD
s=TruncatedSVD(n_components=5, n_iter=7, random_state=42)
sample_feat_vect=s.fit_transform(final_tf_idf_vect)


# In[115]:


sample_feat_vect


# In[138]:


# TF-IDF weighted Word2Vec
tf_idf_features = tf_idf_vect.get_feature_names() # tfidf words/col-names
# final_tf_idf is the sparse matrix with row= sentence, col=word and cell_val = tfidf

tfidf_sent_vectors = [] # the tfidf-w2v for each sentence/review is stored in this list
row=0
for sent in list_of_sentence: # for each review/sentence 
    sent_vec = np.zeros(50) # as word vectors are of zero length
    weight_sum =0 # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        if word in words:
            vec = w2v_model.wv[word]
            # obtain the tf_idfidf of a word in a sentence/review
            tf_idf = final_tf_idf_vect[row, tf_idf_features.index(word)]
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
    if weight_sum != 0:
        sent_vec /= weight_sum
    tfidf_sent_vectors.append(sent_vec)
    row += 1


# In[139]:


tf_vec_avg=np.array(tfidf_sent_vectors)


# In[140]:


std_data=tf_vec_avg


#from sklearn.manifold import TSNE

tmodel=TSNE(n_components=2,random_state=0,perplexity=30,n_iter=1000)
tsne_data=tmodel.fit_transform(std_data)

tsne_data = np.vstack((tsne_data.T, label_data)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("dim1", "dim2", "score"))


sns.FacetGrid(tsne_df, hue="score", size=6).map(plt.scatter, 'dim1', 'dim2').add_legend()
plt.title("TSNE for TF-IDF Word2Vec")
plt.show()

From the above diagrams we can not be able to separate the positive or negative reviews clearly. Even though some of the plots need more working on since sample set size is just 2000