import numpy as np
import pandas as pd

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
            

Artist("Kanye West")


import random, re
# Markov Model
# freqDict is a dict of dict containing frequencies
def addToDict(fileName, freqDict):
	f = open(fileName, 'r')
	words = re.sub("\n", " \n", f.read()).lower().split(' ')

	# count frequencies curr -> succ
	for curr, succ in zip(words[1:], words[:-1]):
		# check if curr is already in the dict of dicts
		if curr not in freqDict:
			freqDict[curr] = {succ: 1}
		else:
			# check if the dict associated with curr already has succ
			if succ not in freqDict[curr]:
				freqDict[curr][succ] = 1;
			else:
				freqDict[curr][succ] += 1;

	# compute percentages
	probDict = {}
	for curr, currDict in freqDict.items():
		probDict[curr] = {}
		currTotal = sum(currDict.values())
		for succ in currDict:
			probDict[curr][succ] = currDict[succ] / currTotal
	return probDict

def markov_next(curr, probDict):
	if curr not in probDict:
		return random.choice(list(probDict.keys()))
	else:
		succProbs = probDict[curr]
		randProb = random.random()
		currProb = 0.0
		for succ in succProbs:
			currProb += succProbs[succ]
			if randProb <= currProb:
				return succ
		return random.choice(list(probDict.keys()))

def makeRap(curr, probDict, T = 50):
	rap = [curr]
	for t in range(T):
		rap.append(markov_next(rap[-1], probDict))
	return " ".join(rap)

if __name__ == '__main__':
	rapFreqDict = {}
	rapProbDict = addToDict('lyrx.txt', rapFreqDict)
	
	startWord = input("Write down a word to start with: ")
	print("Ghost Writer:")
	print(makeRap(startWord, rapProbDict))