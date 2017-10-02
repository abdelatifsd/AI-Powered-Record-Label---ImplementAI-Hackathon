#Main dependencies 
import numpy as np
import pandas as pd

# Self-Organizing Maps visualization tools
from pylab import pcolor,colorbar,bone,plot,show

# Natural Languange Processing library - Used to clean the texts
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#Importing the dataset
dataset = pd.read_csv("openreg_dataset.csv")
text = dataset.iloc[:,2].values
text = list(text)

#Removing NaN (empty spaces) in the list of articles
#Ignore error
for i in range(len(text)):
    if type(text[i]) == float:
        text.pop(i)
        
#Declaring the stemming object
ps = PorterStemmer()

cleaned_text = []

# Loop to clean the split, clean + stem the text, and then rejoin the parts splitted
for i in range(len(text)):
    newver = re.sub('[^a-zA-z]',' ', text[i])
    newver = newver.lower()
    newver = newver.split()
    newver = [ps.stem(word) for word in newver if not word in stopwords.words('english')]
    newver = ' '.join(newver)
    cleaned_text.append(newver)
    
#Creating a sparce matrix to prepare input for the Self-Organizing Maps input nodes.
#Similar to what we did in clustering
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 100000)
X = cv.fit_transform(cleaned_text).toarray()



# Self Organizing Maps
from minisom import MiniSom
# Parameters = SOM grids dimensions(20,20) - Number of input nodes (= to input features) 
# sigma = radius surrounding the best matching unit (Gets smallers as you get closer to convergence)
# Learning rate = how fast the SOM learn the clusters
som = MiniSom(20,20,input_len = 14059, sigma = 1, learning_rate = 0.5 )

# Initializing the weights
# Note: the weights in SOMs represent the characteristics of the nodes
som.random_weights_init(X)
# training the model
#parameters = data and number of iterations
som.train_random(X,1000)

#Building the SOM graph
# The legend meter: 0(black cell) = Mean interneuron distances is really small between the winning node and it's neighbors
# 1(white cell) = Mean interneuron distances is huge between the winning node and it's neighbors
# Usually means that the general rule isn't being followed and that there is an unusual relationship
# between the clusters at hand (sometimes interpreted as an anomaly)
bone()
pcolor(som.distance_map().T)
colorbar()

# The clusters in the best winning nodes (Best matching units)
mappings = som.win_map(X)


#Extracting a specific cluster
cluster = mappings[(6,9)]


#transformering the words back to their original format (numbers to actual text)
cluster_text = cv.inverse_transform(cluster)
















