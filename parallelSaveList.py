from multiprocessing import Process, Manager
from PRADAfunctions import *

def parallelPolMiniLex(sharedLexicon,theDict,theIndex):
    tempMiniLex=getMiniLexicon(sharedLexicon,theIndex)
    tempPolMiniLex=getPolygons4MiniLex(tempMiniLex)
    if tempPolMiniLex == {}:
        return
    theDict[theIndex]=tempPolMiniLex

#for multiprossecing there has to be a main
if __name__ == '__main__':
    manager = Manager()
    myNProc=8

    polyDict = manager.list()
    sharedLexicon = manager.dict()
    
    lexicon=openLexiconFile("lexicon100k.pkl")
    #copy lexicon into sharedLexicon
    for e in lexicon:
        sharedLexicon[e]=lexicon[e]

    #target is the function
    processes = [Process(target=parallelPolMiniLex,args=(sharedLexicon,polyDict,i+1)) for i in range(2)]
    
    # start the processes
    for p in processes:
        p.start()
        
    # Join them so everything is syncronized
    for p in processes:
        p.join()
        
    # for val in polyDict:... curiosly doesn't work
    # for i in range(1,myNProc):
    #     print (i+1, polyDict[i+1])

    # file_name="polMiniLexDictParallel.pkl"
    # fileObject=open(file_name,"wb")
    # #Maybe copy polyDict into a simple dictionary
    # pickle.dump(polyDict,fileObject)
    # fileObject.close()
    
