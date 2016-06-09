from multiprocessing import Process, Manager,cpu_count
from PRADAfunctions import *

def parallelPolMiniLex(lexicon,theList,theIndex):
    tempMiniLex=getMiniLexicon(lexicon,theIndex+1)
    tempPolMiniLex=getPolygons4MiniLex(tempMiniLex)
    if tempPolMiniLex == {}:
        return
    theList[theIndex]=tempPolMiniLex

#for multiprossecing there has to be a main
if __name__ == '__main__':
    manager = Manager()
    myNProc=cpu_count()
    tDN=16 #total number of detectors
    print ("Number of processors = ",myNProc)

    polyList = manager.list()
    for i in range(tDN):
        polyList.append({})
    
    lexicon=openLexiconFile("lexicon100k.pkl")
  
    #target is the function
    processes = [Process(target=parallelPolMiniLex,
                         args=(lexicon,polyList,i)) for i in range(tDN)]
    
    # start the processes
    # for p in processes:
    #     p.start()
        
    # Join them so everything is syncronized
    # for p in processes:
    #     p.join()

    for i in range(8):
        processes[i].start()

    for i in range(8):
        processes[i].join()

    for i in range(8,16):
        processes[i].start()

    for i in range(8,16):
        processes[i].join()

    # for val in polyDict:... curiosly doesn't work
    # for i in range(1,myNProc):
    #     print (i+1, polyDict[i+1])

    file_name="polMiniLexListParallel.pkl"
    fileObject=open(file_name,"wb")
    #Maybe copy polyDict into a simple dictionary
    pickle.dump(list(polyList),fileObject)
    fileObject.close()
