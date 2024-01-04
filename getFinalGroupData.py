"""
Created on October 12 2021
Created by Joshua Fantillo
The University of the Fraser Valley
Comp 440
"""
import pandas as pd 

#gets the csv made in GetGroupResults.py
def getData():
    df_groupData = pd.read_csv(r"GroupData.csv") 
    return df_groupData

#this function cleans up the dataset and saves bagOfWords and bagOfTags seperatly so it can be modified easier
def getBagOf(df_groupData, bagOfTagsHold, bagOfWordsHold):
    df_groupData = pd.read_csv(r"GroupData.csv") 
    df_groupData = df_groupData.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
    bagOfWordsHold.append(df_groupData['Bag of Words'])
    df_groupData = df_groupData.drop(['Bag of Words'], axis =1)
    bagOfTagsHold.append(df_groupData['Bag of Tags'])
    df_groupData = df_groupData.drop(['Bag of Tags'], axis =1)
    return df_groupData, bagOfTagsHold, bagOfWordsHold

#this saves both bag of (words/tags) into their own seperate array and cleans the data
def modifyBagOf(bagOfHold, bagOf):
    x = []
    for i in range(len(bagOfHold[0])):
        bagOfHold[0][i] = bagOfHold[0][i].replace('[', '')
        bagOfHold[0][i] = bagOfHold[0][i].replace(']', '')
        bagOfHold[0][i] = bagOfHold[0][i].replace('(', '')
        bagOfHold[0][i] = bagOfHold[0][i].replace(')', '')
        bagOfHold[0][i] = bagOfHold[0][i].replace('\'', '')
        bagOfHold[0][i] = bagOfHold[0][i].replace(' ', '')
        x = bagOfHold[0][i].split(',')
        bagOf.append(x)   
    return bagOf

#this function gets the average of whichever sound function we send to it (mfcc, log fbank, ssc)
def getAverageSoundFunctions(df_groupData, feature):
    value = []
    valueHold = []
    hold = []
    holdRow=[]
    x = []
    average = []
    total = 0
    for i in range(len(df_groupData[feature])):
        valueHold.append(df_groupData[feature][i])
    #cleans the data
    for i in range(len(valueHold)):
        valueHold[i] = valueHold[i].replace('[', '')
        valueHold[i] = valueHold[i].replace(']', '')
        valueHold[i] = valueHold[i].replace('\n', '')
        valueHold[i] = valueHold[i].replace('...', '')
        #splits the data
        x = valueHold[i].split(' ')
        hold.append(x)
    #saves the cleaned data back into an array only if its not empty
    for i in range(len(hold)):
        for j in range(len(hold[i])):
            if (hold[i][j] != ""):
                holdRow.append(hold[i][j])
        value.append(holdRow)
        holdRow = []
    #gets the average value of the data in the array
    for i in range(len(value)):
        for j in range(len(value[i])):
            total = total + float(value[i][j])
        average.append(total/len(value[i]))
        total = 0
    return average

#reassembles the data back into the csv file and creates a final csv that will be used
def reAssembleData(averageMFCC, averageLogFBank, averageSSC, df_groupData):
    df_groupData = df_groupData.drop(['MFC Features'], axis =1)
    df_groupData['MFCC Features'] = averageMFCC
    df_groupData = df_groupData.drop(['log fBank Features'], axis =1)
    df_groupData['log fBank Features'] = averageLogFBank
    df_groupData = df_groupData.drop(['SSC Features'], axis =1)
    df_groupData['SSC Features'] = averageSSC
    df_groupData = df_groupData.drop(['fBank Features'], axis=1)
    df_groupData = df_groupData.apply(pd.to_numeric)
    df_groupData.to_csv('GroupDataFinal.csv')
    
#runs this file
def runGetFinalGroupData():
    bagOfWordsHold = []
    bagOfTagsHold = []
    bagOfWords = []
    bagOfTags = []
    averageMFCC = []
    averageLogFBank = []
    averageSSC = []
    
    df_groupData = getData()
    df_groupData, bagOfTagsHold, bagOfWordsHold = getBagOf(df_groupData, bagOfTagsHold, bagOfWordsHold)
    print("Cleaning Bag Of Words and Tags")
    bagOfWords = modifyBagOf(bagOfWordsHold, bagOfWords)
    bagOfTags = modifyBagOf(bagOfTagsHold, bagOfTags)
    print("Cleaning Sound Data")
    averageMFCC = getAverageSoundFunctions(df_groupData, 'MFC Features')
    averageLogFBank = getAverageSoundFunctions(df_groupData, 'log fBank Features')
    averageSSC =getAverageSoundFunctions(df_groupData, 'SSC Features')
    print("Saving Final Data CSV File")
    reAssembleData(averageMFCC, averageLogFBank, averageSSC, df_groupData)
    

    
    
    

