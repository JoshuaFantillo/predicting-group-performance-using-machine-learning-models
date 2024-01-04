"""
Created on October  12 2021
Created by Joshua Fantillo
The University of the Fraser Valley
Comp 440
"""
import pandas as pd 


#reads in the Group-Level Meeting Data.csv
def getGroupResults():
    Group = pd.read_csv(r"gap-corpus-master\Final-Corpus-Transcripts-Annotations-Data\Group-Individual-Data\Group-Level Meeting Data.csv")     
    return Group

#reads in the Individual-Level Meeting Data.csv
def getIndData():
    Individual = pd.read_csv(r"gap-corpus-master\Final-Corpus-Transcripts-Annotations-Data\Group-Individual-Data\Individual-Level Meeting Data.csv")
    return Individual

#this seperates the group number and group member and saves it back to the dataframe
def cleanIndData(Individual):
    holdGroupNumber = []
    holdGroupMember = []
    for i in range(len(Individual)):  
        hold = Individual['Group Member'][i].replace('.', ' ')
        hold = hold.split()
        holdGroupNumber.append(hold[0])
        holdGroupMember.append(hold[1])
    Individual.drop('Group Member', axis=1, inplace=True)
    Individual['Group Number'] = holdGroupNumber
    Individual['Group Member'] = holdGroupMember
    return Individual

##calculates the average year of the group memebers in each group
def getAvgYear(Group, Individual):
    averageYear = []
    averageYear = [0 for i in range(28)]
    Individual['Group Number'] = [int(x) for x in Individual['Group Number']]
    Individual['Year at UFV'] = [int(x) for x in Individual['Year at UFV']]
    Group['Meeting Size'] = [int(x) for x in Group['Meeting Size']]
    for i in range(len(Individual)):
        groupNumber = Individual['Group Number'][i]
        year = Individual['Year at UFV'][i]
        groupSize = Group['Meeting Size'][groupNumber-1]
        aveYear = (year/groupSize)
        averageYear[groupNumber-1] = averageYear[groupNumber-1] + aveYear
    return averageYear

#gets the average english score for each group
def getAvgEng(Group, Individual):
    averageEnglish = []
    averageEnglish = [0 for i in range(28)]
    Individual['English'] = [int(x) for x in Individual['English']]
    for i in range(len(Individual)):
        groupNumber = Individual['Group Number'][i]
        english = Individual['English'][i]
        groupSize = Group['Meeting Size'][groupNumber-1]
        aveEnglish = (english/groupSize)
        averageEnglish[groupNumber-1] = averageEnglish[groupNumber-1] + aveEnglish
    return averageEnglish

#we load in the GroupData.csv we created in GetGroupData.py
def getGroupData():
    GroupData = pd.read_csv('GroupData.csv')
    return GroupData

#adds all the values we want into the groupdata.csv file we made so that we can analyze in the next step
def addResults(GroupData, Group, averageYear, averageEnglish):
    GroupData['Avg Year'] = averageYear
    GroupData['Avg English'] = averageEnglish
    GroupData['Meeting Size'] = pd.Series(Group['Meeting Size'])
    GroupData['Meeting Length'] = pd.Series(Group['Meeting Length in Minutes'])
    GroupData['AGS'] = pd.Series(Group['AGS'])
    GroupData['Group_TE'] = pd.Series(Group['Group_TE'])
    GroupData['Group_WW'] = pd.Series(Group['Group_WW'])
    GroupData['Group_TM'] = pd.Series(Group['Group_TM'])
    GroupData['Group_Eff'] = pd.Series(Group['Group_Eff'])
    GroupData['Group_QW'] = pd.Series(Group['Group_QW'])
    GroupData['Group_Sat'] = pd.Series(Group['Group_Sat'])
    return GroupData

#runs this file
def runGetGroupResults():
    Group = getGroupResults()
    Individual = getIndData()
    Individual = cleanIndData(Individual)
    averageYear = getAvgYear(Group, Individual)
    averageEnglish = getAvgEng(Group, Individual)
    GroupData = getGroupData()
    GroupData = addResults(GroupData, Group, averageYear, averageEnglish)
    GroupData.to_csv('GroupData.csv')
    print("Finished Getting Group Results")

