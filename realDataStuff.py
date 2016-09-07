import csv
import pickle
from PRADAfunctions import *
#Pedro's code

def pickleRealData(txtFileName="data.txt", pklName="data.pkl"):
    #new lists
    aList,bList=[],[]
    #open the txt file and put it into a list with strings
    with open(txtFileName) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        for line in tsvreader:
            aList.append(line[:])
    #convert the strings into integers
    for e in aList:
            bList.append([int(i) for i in e])
    #save the shit out of the data in a new thing called pickle
    fileObject=open(pklName,'wb')
    pickle.dump(bList,fileObject)
    fileObject.close()

def unpickleRealData(pklName="data.pkl"):
    pkl_file = open(pklName,'rb')
    dataInList = pickle.load(pkl_file)
    pkl_file.close()
    return dataInList



