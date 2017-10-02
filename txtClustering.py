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

dataset = pd.read_csv("songdata.csv")

artists = (dataset.iloc[:,0])
lyrics = (dataset.iloc[:,3])

artist_index = []

def Artist(x):
    for i in range(len(artists)):
        if artists[i] == x:
            artist_index.append(i)
    
    with open("lyrx2.csv", "w") as csv_file:
        for i in artist_index:
            print(lyrics[i], file=csv_file)

Artist("Drake")            


z = pd.read_csv("lyrx2.csv", error_bad_lines=False)
text = z.iloc[:,0]
text = list(text)



ps = PorterStemmer()

cleaned_text = []

# Creating the clean version
for i in range(len(text)):
    newver = re.sub('[^a-zA-z]',' ', text[i])
    newver = newver.lower()
    newver = newver.split()
    newver = [ps.stem(word) for word in newver if not word in stopwords.words('english')]
    newver = ' '.join(newver)
    cleaned_text.append(newver)
    
#Compare the clean version to the one showcased above
print(cleaned_text)



from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 10000)
X = cv.fit_transform(cleaned_text).toarray()



# Self Organizing Maps
from minisom import MiniSom
# Parameters = SOM grids dimensions(20,20) - Number of input nodes (= to input features) 
# sigma = radius surrounding the best matching unit (Gets smallers as you get closer to convergence)
# Learning rate = how fast the SOM learn the clusters
som = MiniSom(20,20,input_len = X.shape[1], sigma = 0.5, learning_rate = 0.3 )

# Initializing the weights
# Note: the weights in SOMs represent the characteristics of the nodes
#som.random_weights_init(X)
# training the model
#parameters = data and number of iterations
som.train_random(X,1000)


bone()
pcolor(som.distance_map().T)
colorbar()
show()


# The clusters in the best winning nodes (Best matching units)
mappings = som.win_map(X)

#Extracting a specific cluster
cluster = mappings[(0,0)]


#transformering the words back to their original format (numbers to actual text)
cluster_text = cv.inverse_transform(cluster)
print(cluster_text)
# I recommend using spyder IDE as you can easily access/explore all the results saved in the variables you created







"""
import numpy as np
import pandas as pd
import textblob as TextBlob
import csv

dataset = pd.read_csv("songdata.csv")

artists = (dataset.iloc[:,0])
lyrics = (dataset.iloc[:,3])

artist_index = []

def Artist(x):
    for i in range(len(artists)):
        if artists[i] == x:
            artist_index.append(i)
    
    with open("lyrx.txt", "w") as text_file:
        for i in artist_index:
            print(lyrics[i], file=text_file)



with open("out.tsv",'w') as tsvfile:
    fileWriter = csv.writer(tsvfile, delimiter = '\t')
    fileWriter.writerow(["Review"] + ["Liked"])
    for sent in sentences:
         analysis = TextBlob(str(sent))
         x = analysis.sentiment.polarity # we will classify the sentence as positive(1) or negative(0). The classification depends on the polarity
         if x >= 0:
            x = 1
         elif x < 0:
            x = 0
         fileWriter.writerow(sent + [x])
"""        