"""
Created on October 12 2021
Created by Joshua Fantillo
The University of the Fraser Valley
Comp 440
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

#reads in the dataset
def getData():
    df_groupData = pd.read_csv(r"GroupDataFinal.csv") 
    return df_groupData

#different modifications of the data we will preform on the analysis
def getMods(df_groupData, count):
    switcher = {
        0: df_groupData,
        1: df_groupData.drop(['Total Words'], axis=1),
        2: df_groupData.drop(['Total Sentences'],axis=1), 
        3: df_groupData.drop(['Neutral Sentiment'],axis=1),
        4: df_groupData.drop(['Filled Pauses'],axis=1),
        5: df_groupData.drop(['Avg Cos Similarity'],axis=1),
        6: df_groupData.drop(['Type Token Ratio'],axis=1),
        7: df_groupData.drop(['Type Tag Ratio'],axis=1),
        8: df_groupData.drop(['MFCC Features'],axis=1),
        9: df_groupData.drop(['log fBank Features'],axis=1),
        10: df_groupData.drop(['SSC Features'],axis=1),
        11: df_groupData.drop(['Avg Year'],axis=1),
        12: df_groupData.drop(['Avg English'],axis=1),
        13: df_groupData.drop(['Group_TE'],axis=1),
        14: df_groupData.drop(['Group_WW'],axis=1),
        15: df_groupData.drop(['Group_TM'],axis=1),
        16: df_groupData.drop(['Group_Eff'],axis=1),
        17: df_groupData.drop(['Group_Sat'],axis=1),
        18: df_groupData.drop(['Total Words','Total Sentences'],axis=1),
        19: df_groupData.drop(['Type Token Ratio','Type Tag Ratio'],axis=1),
        20: df_groupData.drop(['MFCC Features','log fBank Features','SSC Features'],axis=1),
        21: df_groupData.drop(['Avg Year', 'Avg English','Group_TE','Group_WW','Group_TM','Group_Eff','Group_Sat'],axis=1),
        }
    return switcher.get(count)

#trains the dataset
def getTrainAndTestData(df_groupData):
    X = df_groupData.drop(['AGS'],axis=1)
    y = df_groupData['AGS']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
    return X, y, X_train, X_test, y_train, y_test

#gets the random forest
def getRandomForest(X_train, X_test, y_train, y_test):
    rfMean = [0,0,0,0,0]
    rfAcc = [0,0,0,0,0]
    rfNest = [1,5,9,20,50]
    #uses 5 different n_estimators for testing
    for i in range(len(rfNest)):
        rf = RandomForestRegressor(n_estimators = rfNest[i], random_state = 42)
        #Train the model on training data
        rf.fit(X_train, y_train);
        #Use the forest's predict method on the test data
        predictions = rf.predict(X_test)
        #Calculate the absolute errors
        errors = abs(predictions - y_test)
        # Calculate mean absolute percentage error (MAPE)
        mape = 100 * (errors / y_test)
        # Calculate accuracy
        accuracy = 100 - np.mean(mape)
        rfMean[i] = round(np.mean(errors), 2)
        rfAcc[i] = accuracy
    return rfMean, rfAcc

#gets the decision tree regressor
def getDecTreeReg(X_train, X_test, y_train, y_test):
    dtRegTrain = [0,0,0,0,0]
    dtRegTest = [0,0,0,0,0]
    depth = [1,2,3,5,10]
    #uses 5 different max depths for testing
    for i in range(len(depth)):
        tree = DecisionTreeRegressor(max_depth=depth[i], random_state=0)
        tree.fit(X_train, y_train)
        target_train_predicted = tree.predict(X_train)
        target_test_predicted = tree.predict(X_test)
        dtRegTrain[i] = str(sum(target_train_predicted)/len(target_train_predicted))
        dtRegTest[i] = str(sum(target_test_predicted)/len(target_test_predicted))
    return dtRegTrain, dtRegTest

#compares the gradient boosting regressor with the random forest regressor
def getCompareDecTreeVsRF(X, y):
    compareArray1 = [0,0,0,0,0]
    compareArray2 = [0,0,0,0,0]
    cvList = [1, 2, 5, 10, 50]
    #uses 5 different n_estimators for testing
    for i in range(len(cvList)):
        gradient_boosting = GradientBoostingRegressor(n_estimators=cvList[i])
        cv_results_gbdt = cross_validate(gradient_boosting, X, y, scoring="neg_mean_absolute_error",n_jobs=2,)
        random_forest = RandomForestRegressor(n_estimators=cvList[i], n_jobs=2)
        cv_results_rf = cross_validate(random_forest, X, y, scoring="neg_mean_absolute_error",n_jobs=2,)
        compareArray1[i] = cv_results_gbdt['test_score'].mean()
        compareArray2[i] = cv_results_rf['test_score'].mean()
    return compareArray1, compareArray2
        
#gets the knn 
def getKnn(X_train, X_test, y_train, y_test):
    knnArray = [0,0,0,0,0]
    kValues = [1, 2, 5, 8, 10]
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    #uses 5 different n_neighbours for testing
    for i in range(len(kValues)):
        classifier = KNeighborsClassifier(n_neighbors=kValues[i])
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        average = sum(y_pred)/len(y_pred)
        knnArray[i] = average
        #print(average) 
    return knnArray

#puts all the results into a result csv file
def makeResultsCSV(KNN1,KNN2,KNN5,KNN8,KNN10,CVRGBDT1,CVRGBDT2,CVRGBDT5,CVRGBDT10,CVRGBDT50,CVRRF1,CVRRF2,CVRRF5,CVRRF10,
                   CVRRF50,DTRTRAIN1,DTRTRAIN2,DTRTRAIN3,DTRTRAIN5,DTRTRAIN10,DTRTEST1,DTRTEST2,DTRTEST3,DTRTEST5,
                   DTRTEST10,RFMEAN1,RFMEAN5,RFMEAN9,RFMEAN20,RFMEAN50,RFACC1,RFACC5,RFACC9,RFACC20,RFACC50):
     Results = pd.DataFrame()
     Results['KNN N=1'] = KNN1 
     Results['KNN N=2'] = KNN2
     Results['KNN N=5'] = KNN5
     Results['KNN N=10'] = KNN10
     Results['CVRGBDT N Estimator=1'] = CVRGBDT1
     Results['CVRGBDT N Estimator=2'] = CVRGBDT2
     Results['CVRGBDT N Estimator=5'] = CVRGBDT5
     Results['CVRGBDT N Estimator=10'] = CVRGBDT10
     Results['CVRGBDT N Estimator=50'] = CVRGBDT50
     Results['CVRRF N Estimator=1'] = CVRRF1
     Results['CVRRF N Estimator=2'] = CVRRF2
     Results['CVRRF N Estimator=5'] = CVRRF5
     Results['CVRRF N Estimator=10'] = CVRRF10
     Results['CVRRF N Estimator=50'] = CVRRF50
     Results['DTRTRAIN Depth=1'] = DTRTRAIN1
     Results['DTRTRAIN Depth=2'] = DTRTRAIN2
     Results['DTRTRAIN Depth=3'] = DTRTRAIN3
     Results['DTRTRAIN Depth=5'] = DTRTRAIN5
     Results['DTRTRAIN Depth=10'] = DTRTRAIN10
     Results['DTRTEST Depth=1'] = DTRTEST1
     Results['DTRTEST Depth=2'] = DTRTEST2
     Results['DTRTEST Depth=3'] = DTRTEST3
     Results['DTRTEST Depth=5'] = DTRTEST5
     Results['DTRTEST Depth=10'] = DTRTEST10
     Results['RFMEAN N Estimator=1'] = RFMEAN1
     Results['RFMEAN N Estimator=5'] = RFMEAN5
     Results['RFMEAN N Estimator=9'] = RFMEAN9
     Results['RFMEAN N Estimator=20'] = RFMEAN20
     Results['RFMEAN N Estimator=50'] = RFMEAN50
     Results['RFACC N Estimator=1'] = RFACC1
     Results['RFACC N Estimator=5'] = RFACC5
     Results['RFACC N Estimator=9'] = RFACC9
     Results['RFACC N Estimator=20'] = RFACC20
     Results['RFACC N Estimator=50'] = RFACC50
     
     Results.to_csv('Results.csv')
     
#runs the file
def runDataAnalysis():
    knnArray = []
    compareArray1, compareArray2 = ([] for i in range(2))
    dtRegTrain, dtRegTest = ([] for i in range(2))
    rfMean, rfAcc = ([] for i in range(2))
    KNN1, KNN2, KNN5, KNN8, KNN10 = ([] for i in range(5))
    CVRGBDT1, CVRGBDT2, CVRGBDT5, CVRGBDT10, CVRGBDT50 = ([] for i in range(5))
    CVRRF1, CVRRF2, CVRRF5, CVRRF10, CVRRF50 = ([] for i in range(5))
    DTRTRAIN1, DTRTRAIN2, DTRTRAIN3, DTRTRAIN5, DTRTRAIN10 = ([] for i in range(5))
    DTRTEST1, DTRTEST2, DTRTEST3, DTRTEST5, DTRTEST10 = ([] for i in range(5))
    RFMEAN1, RFMEAN5, RFMEAN9, RFMEAN20, RFMEAN50 = ([] for i in range(5))
    RFACC1, RFACC5, RFACC9, RFACC20, RFACC50 = ([] for i in range(5))
    
    df_groupData = getData()
    print("Preforming Analysis on: ")
    for i in range(0,22):
        print("Modification " + str(i))
        #preforms for all different modifications of the data
        df_groupDataTest = getMods(df_groupData, i)
        X, y, X_train, X_test, y_train, y_test = getTrainAndTestData(df_groupDataTest)
        rfMean, rfAcc = getRandomForest(X_train, X_test, y_train, y_test)
        dtRegTrain, dtRegTest = getDecTreeReg(X_train, X_test, y_train, y_test)
        compareArray1, compareArray2 = getCompareDecTreeVsRF(X, y)
        knnArray = getKnn(X_train, X_test, y_train, y_test)
        KNN1.append(knnArray[0])
        KNN2.append(knnArray[1])
        KNN5.append(knnArray[2])
        KNN8.append(knnArray[3])
        KNN10.append(knnArray[4])
        CVRGBDT1.append(compareArray1[0])
        CVRGBDT2.append(compareArray1[1])
        CVRGBDT5.append(compareArray1[2])
        CVRGBDT10.append(compareArray1[3])
        CVRGBDT50.append(compareArray1[4])
        CVRRF1.append(compareArray2[0])
        CVRRF2.append(compareArray2[1])
        CVRRF5.append(compareArray2[2])
        CVRRF10.append(compareArray2[3])
        CVRRF50.append(compareArray2[4])
        DTRTRAIN1.append(dtRegTrain[0])
        DTRTRAIN2.append(dtRegTrain[1])
        DTRTRAIN3.append(dtRegTrain[2])
        DTRTRAIN5.append(dtRegTrain[3])
        DTRTRAIN10.append(dtRegTrain[4])
        DTRTEST1.append(dtRegTest[0])
        DTRTEST2.append(dtRegTest[1])
        DTRTEST3.append(dtRegTest[2])
        DTRTEST5.append(dtRegTest[3])
        DTRTEST10.append(dtRegTest[4])
        RFMEAN1.append(rfMean[0])
        RFMEAN5.append(rfMean[1])
        RFMEAN9.append(rfMean[2])
        RFMEAN20.append(rfMean[3])
        RFMEAN50.append(rfMean[4])
        RFACC1.append(rfAcc[0])
        RFACC5.append(rfAcc[1])
        RFACC9.append(rfAcc[2])
        RFACC20.append(rfAcc[3])
        RFACC50.append(rfAcc[4])
    
    print("Saving Results of Analysis")
    makeResultsCSV(KNN1,KNN2,KNN5,KNN8,KNN10,CVRGBDT1,CVRGBDT2,CVRGBDT5,CVRGBDT10,CVRGBDT50,CVRRF1,CVRRF2,CVRRF5,CVRRF10,
                   CVRRF50,DTRTRAIN1,DTRTRAIN2,DTRTRAIN3,DTRTRAIN5,DTRTRAIN10,DTRTEST1,DTRTEST2,DTRTEST3,DTRTEST5,
                   DTRTEST10,RFMEAN1,RFMEAN5,RFMEAN9,RFMEAN20,RFMEAN50,RFACC1,RFACC5,RFACC9,RFACC20,RFACC50)
        
        
        
