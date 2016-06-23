from multiprocessing import Process, Manager,cpu_count
from PRADAfunctions import *

def parallelPolMiniLex(lexicon,theList,theIndex,alpha):
    tempMiniLex=getMiniLexicon(lexicon,theIndex+1)
    tempPolMiniLex=getPolygons4MiniLex(tempMiniLex,alpha=alpha)
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
    
    lexicon=openLexicon("lexicon1MNEc.pkl")
    #if number of points=10k use alpha=0.01, or
    #if number of points=100k use alpha=0.1, or
    #if number of points=1M use alpha=1.0

    alpha=[1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]

    #target is the function
    processes = [Process(target=parallelPolMiniLex,
                         args=(lexicon,polyList,i,alpha[i])) for i in range(tDN)]
    
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

    file_name="polMiniLexListParallel1MNEcSpecialAlpha.pkl"
    fileObject=open(file_name,"wb")
    #Maybe copy polyDict into a simple dictionary
    pickle.dump(list(polyList),fileObject)
    fileObject.close()
