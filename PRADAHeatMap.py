from realDataStuff import *
from PRADAfunctions import *
import sys
import os.path

txtfile=sys.argv[1]
polMiniLexList=sys.argv[2]
detectOrder=int(sys.argv[3])
threshold=int(sys.argv[4])

def PRADAHeatMap(txtfile,polMiniLexList,detectOrder,threshold):
    print ("Opening polMiniLexList")
    polMiniLexList=openPolMiniLexList(polMiniLexList)
    pklName=txtfile.split(".")[0]+".pkl"
    if not os.path.exists(pklName):
        print ("Making pkl file")
        pickleRealData(txtfile,pklName)
    
    print ("Loading exp data")
    rawExpDat=unpickleRealData(pklName)
    print ("len(rawExpDat) = ", len(rawExpDat))
    #print ("Creating subList")
    #expDat=[e[2:] for e in rawExpDat] #if e[1] > 50]
    #print ("len(expDat) =",len(expDat))
    print ("Getting the data over the threshold")
    discDetList=getDiscriminateDetectList(rawExpDat,threshold,detectOrder)
    print ("Getting the count dict for plotting")
    countDict=getCountDict(discDetList)
    print ("Getting the right miniPol for plotting the heatmap")
    miniPol=polMiniLexList[detectOrder-1]
    print ("FinalLy doing the plot!")
    plotCountDictPolygons(countDict,miniPol)
    maxCounts=getMaxCount(countDict)
    print ("Max Counts= ", maxCounts)
    plt.show()

PRADAHeatMap(txtfile,polMiniLexList,detectOrder,threshold)