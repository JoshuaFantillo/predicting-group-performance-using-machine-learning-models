"""
Created on October 12 2021
Created by Joshua Fantillo
The University of the Fraser Valley
Comp 440
"""

""""
runGetGroupData took roughly 17 minutes in testing 
getting sound data take 90% of that times
runDataAnalysis took roughly 4 minutes
total time to run while testing was 21:04
"""
from GetGroupData import runGetGroupData
from GetGroupResults import runGetGroupResults
from getFinalGroupData import runGetFinalGroupData
from DataAnalysis import runDataAnalysis

#project gets run from here
def main():
    print("Running GetGroupData.py")
    runGetGroupData()
    print("Running GetGroupResults.py")
    runGetGroupResults()
    print("Running getFinalGroupData.py")
    runGetFinalGroupData()
    print("Running DataAnalysis.py")
    runDataAnalysis()
    print("Finished Running")

if __name__ == "__main__":
    main()