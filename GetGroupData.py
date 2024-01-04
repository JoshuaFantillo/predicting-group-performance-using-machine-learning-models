"""
Created on October 12 2021
Created by Joshua Fantillo
The University of the Fraser Valley
Comp 440
"""


import pandas as pd 
import nltk
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.probability import FreqDist
from python_speech_features import *


#function gets all the sentences in the transcript and parses them
def getParsed(sentence):
    parsedSentences = []
    for i in range(len(sentence)):
       #gets the sentence
        text = sentence[i]
        #parses the sentence and saves it to tokens
        tokens = nltk.word_tokenize(text)
        #gets rid of the "" at the beggining and end of the sentence
        tokens = tokens[1:-1]
        #adds the parsed sentence to the list
        parsedSentences.append(tokens)
    return parsedSentences

#function gets the total words and sentences from the transcripts
def getTotals(parsedSentences, sentence):
    #used to get total words and total amount of sentences
    #filters out blank sentences
    totalWords = 0
    totalSentences = 0
    for i in range(len(parsedSentences)):
        if(len(parsedSentences[i]) == 1):
            if(parsedSentences[i][0] == '$'):
                totalSentences = totalSentences + 1
        for j in range(len(parsedSentences[i])):
            if (parsedSentences[i][j] != '$'):
                totalWords = totalWords + 1       
    #gets the total sentences wihtout the empty sentences
    totalSentences = len(sentence) - totalSentences
    #gets the average words per sentence and saves it all into an array
    averageWordsPerSentence = totalWords/totalSentences
    return totalWords, totalSentences, averageWordsPerSentence

#function makes the sentence a string
def getSentString(sentence):
    sentence = sentence.astype('str')
    return sentence

#function gets the polarity of the specific text
def get_polarity(text):
    #we use textblob to get a sentiment analysis of our sentences
    return TextBlob(text).sentiment.polarity

#function gets the polarity of the sentence and decides if it is positive, negative, or neutral
def getPolarity(sentence):
    sentence = getSentString(sentence)
    #gets the polarity
    polarity = sentence.apply(get_polarity)
    #decides if the sentence is positive, negative, or neutral
    sentimentType = []
    for i in range(len(sentence)):
       if (polarity[i]>0):
           sentimentType.append('Positive') 
       if (polarity[i] == 0):
           sentimentType.append('Neutral')
       if (polarity[i] < 0):
           sentimentType.append('Negative') 
    positive = 0
    negative = 0
    neutral = 0
    #gets a count for number of positive, negative, and neutral sentences
    for i in range(len(sentence)):
        if(sentimentType[i] == 'Positive'):
            positive = positive + 1
        if(sentimentType[i] == 'Negative'):
            negative = negative + 1
        if(sentimentType[i] == 'Neutral'):
            neutral = neutral + 1
    sentimentCounts = []
    sentimentCounts.append(positive)
    sentimentCounts.append(neutral)
    sentimentCounts.append(negative)
    #graph the sentiment analysis for visual reference       
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    langs = ['Positive', 'Neutral', 'Negative']
    ax.bar(langs,sentimentCounts)
    plt.show()
    #gets the percentages of each positive, negative, and neutral sentiments
    positivePercent = sentimentCounts[0]/len(sentence)
    neutralPercent = sentimentCounts[1]/len(sentence) 
    negativePercent = sentimentCounts[2]/len(sentence)
    return positivePercent, negativePercent, neutralPercent

#function gets the number of pauses in the transcript
def getPauses(sentence, parsedSentences):
    #gets the number of pauses 'uh' or 'um'
    pauseCounter = 0
    for i in range(len(parsedSentences)):
        for j in range(len(parsedSentences[i])):
            if(parsedSentences[i][j] == 'uh' or parsedSentences[i][j]== 'um'):
                pauseCounter = pauseCounter +1
    return pauseCounter

#function gets rid of the stop words in the sentences
def getStopWords(sentence, parsedSentences):
    stop_words=set(stopwords.words("english"))
    #we remove the stop words from the parsed list and save it back to the list
    hold = []
    for i in range(len(parsedSentences)):
        for w in parsedSentences[i]:
            if w not in stop_words:
                hold.append(w)
        parsedSentences[i] = hold
        hold = []
    return parsedSentences
    
#function lemmatizes the words in the sentences
def getLem(sentence, parsedSentences):
    lem = WordNetLemmatizer()
    #we do this to Lemmatize the words into the root words to simplify the process of getting the frequencies of the words
    for i in range(len(parsedSentences)):
        for j in range(len(parsedSentences[i])):
            parsedSentences[i][j] = lem.lemmatize(parsedSentences[i][j],"v")
    return parsedSentences

#function gets the cosine similarity of each sentence and the average similarity for the whole transcript
def getCosSim(parsedSentences):
    #this gets the average cosine similarity of each groups
    average = 0
    #it does 3 sentences neighbouring sentences at a time
    for i in range(len(parsedSentences)-2):
        #skips empty sentences
        hold = i
        while(str(parsedSentences[i]) == "['$']" and (i)<len(parsedSentences)-3):
            i += 1
        sentence1 = str(parsedSentences[i])
        while(str(parsedSentences[i+1]) == "['$']" and (i+1)<len(parsedSentences)-2):
            i += 1
        sentence2 = str(parsedSentences[i+1])
        while(str(parsedSentences[i+2]) == "['$']" and (i+2)<len(parsedSentences)-1):
            i += 1
        sentence3 = str(parsedSentences[i+2])
        i=hold
        documents = [sentence1, sentence2, sentence3]
        count_vectorizer = CountVectorizer()
        sparse_matrix = count_vectorizer.fit_transform(documents)
        #saves it into a pandas dataframe
        doc_term_matrix = sparse_matrix.todense()
        df = pd.DataFrame(doc_term_matrix, columns=count_vectorizer.get_feature_names(), index=['sentence1', 'sentence2', 'sentence3'])
        #gets the cosine similarity for all 3 sentences in a 3x3 matrix
        cosSim = cosine_similarity(df, df)
        total = 0
        #gets the average similarity of that matrix
        for i in range(len(cosSim)):
            for j in range(len(cosSim[i])):
                total = total + cosSim[i][j]
            average = average + total/9
    #calculates the average cosine similarity for the whole transcript
    averageCosSimilarity = average/(len(parsedSentences)-2)
    return averageCosSimilarity

#function gets the word frequencies
def getFreqWords(sentence, parsedSentences):
    #this function gets the frequency of all the words used
    total = FreqDist('')
    #adds up all the words and their frequencies
    for i in range(len(parsedSentences)):
        fdist = FreqDist(parsedSentences[i])
        total = total + fdist
    #gets total amount of words used    
    totalUniqueWords = len(total)
    #plots the top 30 words used
    #should be noted that '$' are pauses in the transcript
    total.plot(30,cumulative=False)
    plt.show()
    return totalUniqueWords, total

#function gets the POS tags
def getPOSTags(sentence, parsedSentences):
    tokenSentence = []
    #POS tagging the parsed sentences 
    for i in range(len(parsedSentences)):
        tokenSentence.append(nltk.pos_tag(parsedSentences[i]))
    return tokenSentence
 
#function gets the bag of words
def getBagOfWords(totalUniqueWords, tokenSentence, total, totalWords):
    #seperates the words and tags to get bagOfWords and bagOfTags
    words = []
    POS = []
    for i in range(len(tokenSentence)):
        for j in range(len(tokenSentence[i])):
            word, pos = tokenSentence[i][j]
            words.append(word)
            POS.append(pos)
    bagOfWords = total.most_common(100)
    #gets the type token ratio for words
    typeTokenRatio = totalUniqueWords/totalWords
    return bagOfWords, typeTokenRatio, POS

#function gets the bag of tags
def getBagOfTags(POS):
    #gets the frequency distribution of the POS so we can get the bagOfTags
    totalPOS = FreqDist('')
    fdistPOS = FreqDist(POS)
    totalPOS = totalPOS + fdistPOS
    #gets total amount of tags used    
    totalUniqueTags = len(totalPOS)
    #gets the top 15 most common tags for bagOfTags
    bagOfTags = totalPOS.most_common(15)
    #should be noted that '$' are pauses in the transcript
    totalPOS.plot(27,cumulative=False)
    plt.show()
    #gets the type tag ratio
    typeTagRatio = totalUniqueTags/len(POS)
    return bagOfTags, typeTagRatio

#function gets all the group data needed
def getGroupData(totalWordsAll, totalSentencesAll, averageWordsPerSentenceAll, positivePercentAll , neutralPercentAll, negativePercentAll,
    pauseCounterAll, averageCosSimilarityAll, bagOfWordsAll, typeTokenRatioAll, bagOfTagsAll, typeTagRatioAll):
    for i in range(0,28):
        #reads in the group data csv
        df = pd.read_csv(r"gap-corpus-master\Final-Corpus-Transcripts-Annotations-Data\Merged\No-Punctuation\CSV Group "+str(i+1)+".csv") 
        sentence = df['Sentence']
        #gets the data for all groups
        parsedSentences = getParsed(sentence)
        totalWords, totalSentences, averageWordsPerSentence = getTotals(parsedSentences, sentence)
        positivePercent, negativePercent, neutralPercent = getPolarity(sentence)
        pauseCounter = getPauses(sentence, parsedSentences)
        parsedSentences = getStopWords(sentence, parsedSentences)
        parsedSentences = getLem(sentence, parsedSentences)
        averageCosSimilarity = getCosSim(parsedSentences)
        totalUniqueWords, total = getFreqWords(sentence, parsedSentences)
        tokenSentence = getPOSTags(sentence, parsedSentences)
        bagOfWords, typeTokenRatio, POS = getBagOfWords(totalUniqueWords, tokenSentence, total, totalWords)
        bagOfTags, typeTagRatio = getBagOfTags(POS)
        #saves it to an array
        totalWordsAll.append(totalWords)
        totalSentencesAll.append(totalSentences)
        averageWordsPerSentenceAll.append(averageWordsPerSentence)
        positivePercentAll.append(positivePercent)
        neutralPercentAll.append(neutralPercent)
        negativePercentAll.append(negativePercent)
        pauseCounterAll.append(pauseCounter)
        averageCosSimilarityAll.append(averageCosSimilarity)
        bagOfWordsAll.append(bagOfWords)
        typeTokenRatioAll.append(typeTokenRatio)
        bagOfTagsAll.append(bagOfTags)
        typeTagRatioAll.append(typeTagRatio)
    return totalWordsAll, totalSentencesAll, averageWordsPerSentenceAll, positivePercentAll , neutralPercentAll, negativePercentAll, pauseCounterAll, averageCosSimilarityAll, bagOfWordsAll, typeTokenRatioAll, bagOfTagsAll, typeTagRatioAll

#this creates a csv file to save all the data found
def getDataFrame(totalWordsAll, totalSentencesAll, averageWordsPerSentenceAll, positivePercentAll , neutralPercentAll, negativePercentAll,
    pauseCounterAll, averageCosSimilarityAll, bagOfWordsAll, typeTokenRatioAll, bagOfTagsAll, typeTagRatioAll, mfccFeaturesAll, fbankFeaturesAll, 
    logfbankFeaturesAll, sscFeaturesAll):
    # Creates pandas DataFrame.  
    GroupData = pd.DataFrame()  
    #adds the data calculated to a pandas DataFrame
    GroupData['Total Words'] = totalWordsAll
    GroupData['Total Sentences'] = totalSentencesAll
    GroupData['Avg Words Per Sentence'] = averageWordsPerSentenceAll
    GroupData['Positive Sentiment'] = positivePercentAll
    GroupData['Negative Sentiment'] = negativePercentAll
    GroupData['Neutral Sentiment'] = neutralPercentAll
    GroupData['Filled Pauses'] = pauseCounterAll
    GroupData['Avg Cos Similarity'] = averageCosSimilarityAll
    GroupData['Bag of Words'] = bagOfWordsAll
    GroupData['Bag of Tags'] = bagOfTagsAll
    GroupData['Type Token Ratio'] = typeTokenRatioAll
    GroupData['Type Tag Ratio'] = typeTagRatioAll
    GroupData['MFC Features'] = mfccFeaturesAll
    GroupData['fBank Features'] = fbankFeaturesAll
    GroupData['log fBank Features'] = logfbankFeaturesAll
    GroupData['SSC Features'] = sscFeaturesAll
    #renames the rows to the group numbers
    for i in range(0,28):
        GroupData = GroupData.rename(index={i: 'Group ' + str(i+1)})
    #saves the csv
    GroupData.to_csv('GroupData.csv')

#gets all the sound data from the corpus
def getSoundData(mfccFeaturesAll, fbankFeaturesAll, logfbankFeaturesAll, sscFeaturesAll):
    for i in range(0,28):
        print("Getting Sound Data For Group: " + str(i+1))
        #reads in the wav file
        (rate,sig) = wav.read("GroupWavFiles\Group " + str(i+1) + ".wav")
        #gets the mfcc features
        mfcc_feat = mfcc(sig,rate, nfft = 1500)
        #gets the fbank features
        fbank_feature = fbank(sig, rate, nfft=1500)
        #gets the log fbank features
        logfbank_feature = logfbank(sig, rate, nfft=1500)
        #gets the ssc features
        ssc_feature = ssc(sig, rate, nfft=1500)
        #saves them to an array
        mfccFeaturesAll.append(mfcc_feat)
        fbankFeaturesAll.append(fbank_feature)
        logfbankFeaturesAll.append(logfbank_feature)
        sscFeaturesAll.append(ssc_feature)
    return mfccFeaturesAll, fbankFeaturesAll, logfbankFeaturesAll, sscFeaturesAll

#runs this file to get all the group data
def runGetGroupData():
    totalWordsAll = []
    totalSentencesAll = []
    positivePercentAll = []
    averageWordsPerSentenceAll = []
    neutralPercentAll = []
    negativePercentAll = []
    pauseCounterAll = []
    averageCosSimilarityAll = []
    bagOfWordsAll = []
    typeTokenRatioAll = []
    bagOfTagsAll = []
    typeTagRatioAll = []
    mfccFeaturesAll = []
    fbankFeaturesAll = []
    logfbankFeaturesAll = []
    sscFeaturesAll = []
    print("Getting Transcript Features")
    totalWordsAll, totalSentencesAll, averageWordsPerSentenceAll, positivePercentAll , neutralPercentAll, negativePercentAll,pauseCounterAll, averageCosSimilarityAll, bagOfWordsAll, typeTokenRatioAll, bagOfTagsAll, typeTagRatioAll = getGroupData(totalWordsAll, totalSentencesAll, averageWordsPerSentenceAll, positivePercentAll , neutralPercentAll, negativePercentAll,
    pauseCounterAll, averageCosSimilarityAll, bagOfWordsAll, typeTokenRatioAll, bagOfTagsAll, typeTagRatioAll)
    print("Finished Getting Transcript Features")
    print("Getting Sound Features")
    mfccFeaturesAll, fbankFeaturesAll, logfbankFeaturesAll, sscFeaturesAll = getSoundData(mfccFeaturesAll, fbankFeaturesAll, logfbankFeaturesAll, sscFeaturesAll)
    print("Finished Getting Sound Features")
    print("Saving To CSV File")
    getDataFrame(totalWordsAll, totalSentencesAll, averageWordsPerSentenceAll, positivePercentAll , neutralPercentAll, negativePercentAll,
    pauseCounterAll, averageCosSimilarityAll, bagOfWordsAll, typeTokenRatioAll, bagOfTagsAll, typeTagRatioAll, mfccFeaturesAll, 
    fbankFeaturesAll, logfbankFeaturesAll, sscFeaturesAll)
    
    
    



