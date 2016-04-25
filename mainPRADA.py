import matplotlib.pyplot as plt
from shapely.geometry import Polygon,MultiPoint,Point
from shapely.ops import cascaded_union
import random
from math import *
from matplotlib import colors



#define los puntos iniciales, el numero de puntos, el valor A, la posicion del detector
def myCurve(x0=0,y0=0,np=50,A=10,S=1,side="lower"):
    def getR(A,S,ang,gamma):
        return (A/S*sin(ang))**(1.0/gamma)
    points=[]
    
    dang=pi/np
    ang=0
    gamma=2.0
    
    
    #Para el barrido angular de los detectores de la parte inferior
    if side=="lower":
        for i in range(np):
            r=getR(A,S,ang,gamma)
            points.append((x0+r*cos(ang),y0+r*sin(ang)))
            ang += dang


    #Para el barrido angular de los detectores de la parte izquierda
    elif side=="left":
        for i in range(np):
            r=getR(A,S,ang,gamma)
            points.append((x0+r*sin(ang),y0+r*cos(ang)))
            ang += dang


    #Para el barrido angular de los detectores de la parte derecha
    elif side=="right":
        for i in range(np):
            r=getR(A,S,ang,gamma)
            points.append((x0-r*sin(ang),y0+r*cos(ang)))
            ang += dang

    #Para el barrido angular de los detectores de la parte superior
    elif side=="upper":
        for i in range(np):
            r=getR(A,S,ang,gamma)
            points.append((x0+r*cos(ang),y0-r*sin(ang)))
            ang += dang

    #Para el barrido angular de los detectores de esquina superior izquierda
    elif side=="ulc":
        for i in range(np):
            r=getR(A,S,ang,gamma)
            points.append((x0+r*cos(ang-pi/4.),y0-r*sin(ang-pi/4.)))
            ang += dang

    #Para el barrido angular de los detectores de esquina superior derecha
    elif side=="urc":
        for i in range(np):
            r=getR(A,S,ang,gamma)
            points.append((x0+r*cos(ang+pi/4.),y0-r*sin(ang+pi/4.)))
            ang += dang

    #Para el barrido angular de los detectores de esquina inferior izquierda
    elif side=="llc":
        for i in range(np):
            r=getR(A,S,ang,gamma)
            points.append((x0+r*cos(ang-pi/4.),y0+r*sin(ang-pi/4.)))
            ang += dang

    #Para el barrido angular de los detectores de esquina inferior derecha
    elif side=="lrc":
        for i in range(np):
            r=getR(A,S,ang,gamma)
            points.append((x0+r*cos(ang+pi/4.),y0+r*sin(ang+pi/4.)))
            ang += dang

    return points


#list of the photomultiplier tubes
dList=[(79.35,0,"lower","on"),
       (49.35,0,"lower","on"),
       (34.35,0,"lower","on"),
       (2.5,2.5,"llc","on"),#corner
       (0,35,"left","on"),
       (2.5,67.5,"ulc","on"),#corner
       (34.35,70,"upper","on"),
       (49.35,70,"upper","on"),
       (79.35,70,"upper","on"),
       (109.35,70,"upper","on"),
       (124.35,70,"upper","on"),
       (156.35,67.5,"urc","on"),#corner
       (158.7,35,"right","on"),
       (156.2,2.5,"lrc","on"),#corner
       (124.35,0,"lower","on"),
       (109.35,0,"lower","on")]


#coordinates of Mondes plate
plateList=[(0,5),
       (5,0),
       (153.7,0),
       (158.7,5),
       (158.7,65),
       (153.7,70),
       (5,70),
       (0,65)]

platePolygon=Polygon(plateList)

#checking if the point is inside the plate

def checkIfInPlate(platePolygon,x,y):

    point=Point(x,y)

    return platePolygon.contains(point)

#giving a random point within the plate
def getRandomInPlate(platePolygon):
    x=random.uniform(0,158.7)
    y=random.uniform(0,70)
    

    while not checkIfInPlate(platePolygon,x,y):
        x=random.uniform(0,158.7)
        y=random.uniform(0,70)

    return x,y

#checking which detector recieves the max signal given a point
def getMaxDetector(dList,x,y):
    sList=getSignals(dList,x,y)

    return sList.index(max(sList))

#checking which detector recieves the second max signal given a point
def get2MaxDetector(dList,x,y):
    sList=getSignals(dList,x,y)

    sList[sList.index(max(sList))]=0

    return sList.index(max(sList))

#generating a list, ordering the max signals
def getMaxList(dList,x,y):
    sList=getSignals(dList,x,y)
    
    return sorted(range(len(sList)), key=lambda k: sList[k], reverse=True)

#getting the points of the max signal for photomultipliers
def getMaxRegions(dList,platePolygon,N):

    maxRegions=[]
    
    for i in range(len(dList)):
        maxRegions.append([])
    

    for i in range(N):
        x,y=getRandomInPlate(platePolygon)

        maxDect=getMaxDetector(dList,x,y)
        maxRegions[maxDect].append((x,y))

    return maxRegions

#getting the points where a detector is max and the next one, second max
def getFirstAndSecMaxPoints(maxRegPoints,firstMaxDect,secMaxDect):
    firstPoints=maxRegPoints[firstMaxDect]

    secondPoints=[]

    for p in firstPoints:
        x,y=p
        if get2MaxDetector(dList,x,y)==secMaxDect:
            secondPoints.append(p)

    return secondPoints

def plotList(miniList):

    xPoints=[e[0] for e in miniList]
    yPoints=[e[1] for e in miniList]

    plt.plot(xPoints,yPoints,"ro")

#plotting the points where the first dect is max and te second dect is second max
def plotFirstSecondMax(maxRegPoints,firstMaxDect,secMaxDect,colorAndForm):
    secPoints=getFirstAndSecMaxPoints(maxRegPoints,firstMaxDect,secMaxDect)

    xPoints=[e[0] for e in secPoints]
    yPoints=[e[1] for e in secPoints]

    plt.plot(xPoints,yPoints,colorAndForm)

#plotting the max region
def plotMaxRegions(dList,platePolygon,N):

    maxRegions=getMaxRegions(dList,platePolygon,N)
    
    for i in range(len(maxRegions)):
        xVals=[e[0] for e in maxRegions[i]]
        yVals=[e[1] for e in maxRegions[i]]
    
        if i==0:
            plt.plot(xVals,yVals,"ro")

        if i==1:
            plt.plot(xVals,yVals,"bo")
                
        if i==2:
            plt.plot(xVals,yVals,"go")
                
        if i==3:
            plt.plot(xVals,yVals,"yo")
                
        if i==4:
            plt.plot(xVals,yVals,"r^")
                
        if i==5:
            plt.plot(xVals,yVals,"bo")

        if i==6:
            plt.plot(xVals,yVals,"y^")
    
        if i==7:
            plt.plot(xVals,yVals,"ro")
    
        if i==8:
            plt.plot(xVals,yVals,"b^")
    
        if i==9:
            plt.plot(xVals,yVals,"g^")
    
        if i==10:
            plt.plot(xVals,yVals,"c^")
    
        if i==11:
            plt.plot(xVals,yVals,"y^")
    
        if i==12:
            plt.plot(xVals,yVals,"ro")

        if i==13:
            plt.plot(xVals,yVals,"go")
    
        if i==14:
            plt.plot(xVals,yVals,"bo")

        if i==15:
            plt.plot(xVals,yVals,"yo")


def plotGeometry():
    
    A=10
    polypoints=100

    dPoints=[] #detector points
    convexPoints=[]
    
    polygons=[]
    polyCoords=[]
    px,py=[],[]
    pltPoly=[]
    
    counter=0

    sList=getSignals(dList,80,55)
    
    
    for e in dList:
        
        S=sList[counter]
        
        print counter,S

        S=S-0.1*S
        dPoints.append(myCurve(e[0],e[1],polypoints,A,S,e[2]))
        polygons.append(0)
        polyCoords.append(0)
        px.append(0)
        py.append(0)
        pltPoly.append(0)
        convexPoints.append(0)
    
        
        counter+=1

    for i in range(len(dList)):
    
        convexPoints[i]=list(MultiPoint(dPoints[i]).convex_hull.exterior.coords)

        #generating the polygons properly

        polygons[i]=Polygon(convexPoints[i])

        polyCoords[i]=list(polygons[i].exterior.coords)

        if dList[i][3]=="off":
            continue

        px[i]=[x[0] for x in polyCoords[i]]
        py[i]=[y[1] for y in polyCoords[i]]

        plt.plot(px[i],py[i],'r')

        pltPoly[i]=plt.Polygon(polyCoords[i], fc="b")
        plt.gca().add_patch(pltPoly[i])


    interPol=polygons[0]
    counter = 0
    for p in polygons:
        if dList[counter][3]=="on":
#            print "dList val", dList[counter][3]
            interPol=interPol.intersection(p)
        counter+=1
#        centroidCoord=list(interPol.centroid.coords)[0]
        interPolList=list(interPol.exterior.coords)

    centroidCoord=list(interPol.centroid.coords)[0]
    print "Centroid = ",centroidCoord

    AreaPol=interPol.area
    
    print "Area Intersection= ",AreaPol

    pIx=[x[0] for x in interPolList]
    pIy=[y[1] for y in interPolList]

#    plt.plot(pIx,pIy,'go')

    pltPolyIntersect = plt.Polygon(interPolList, fc='g')
    
    plt.gca().add_patch(pltPolyIntersect)


#    if polygon2.intersects(polygon1):
#        print "They intersect!!"
#    else:
#        print "They dont intersect!!"

#        return


#    interPol=polygon1.intersection(polygon2)
#    centroidCoord=list(interPol.centroid.coords)[0]
#    interPolList=list(interPol.exterior.coords)

#    p3x=[x[0] for x in interPolList]
#    p3y=[y[1] for y in interPolList]
    
    #plt.plot(p3x,p3y,'go')

#    pltPolyIntersect = plt.Polygon(interPolList, fc='g')

#    plt.gca().add_patch(pltPolyIntersect)

    #p1Centroid=list(polygon1.centroid.coords)[0]
    #plt.plot(p1Centroid[0], p1Centroid[1], "rs")


#    pInterCentroid=list(interPol.centroid.coords)[0]
#    plt.plot(pInterCentroid[0], pInterCentroid[1], "rs")


#    plt.xlim(-200,200)
#    plt.ylim(-200,200)

#myCurve(5)


#Plotting MONDEs perimeter
def plotDetector(dList):
    plt.plot([5,153.7],[0,0],"y",linewidth=2.0)
    plt.plot([153.7,158.7],[0,5],"y",linewidth=2.0)
    plt.plot([158.7,158.7],[5,65],"y",linewidth=2.0)
    plt.plot([158.7,153.7],[65,70],"y",linewidth=2.0)
    plt.plot([153.7,5],[70,70],"y",linewidth=2.0)
    plt.plot([5,0],[70,65],"y",linewidth=2.0)
    plt.plot([0,0],[65,5],"y",linewidth=2.0)
    plt.plot([0,5],[5,0],"y",linewidth=2.0)

#Generating each photomultiplier tube

    for e in dList:
        if e[3]=="off":
            continue
        
        if e[2]=="lower":
            plt.plot([e[0],e[0]],[e[1]-2,e[1]-17],"k",linewidth=10.0)
        
        if e[2]=="upper":
            plt.plot([e[0],e[0]],[e[1]+2,e[1]+17],"k",linewidth=10.0)
        
        if e[2]=="right":
            plt.plot([e[0]+2,e[0]+17],[e[1],e[1]],"k",linewidth=10.0)
        
        if e[2]=="left":
            plt.plot([e[0]-2,e[0]-17],[e[1],e[1]],"k",linewidth=10.0)
        
        if e[2]=="llc":
            plt.plot([e[0]-1.41,e[0]-10.61],[e[1]-1.41,e[1]-10.61],"k",linewidth=10.0)
        
        if e[2]=="ulc":
            plt.plot([e[0]-1.41,e[0]-10.61],[e[1]+1.41,e[1]+10.61],"k",linewidth=10.0)
        
        if e[2]=="lrc":
            plt.plot([e[0]+1.41,e[0]+10.61],[e[1]-1.41,e[1]-10.61],"k",linewidth=10.0)
        
        if e[2]=="urc":
            plt.plot([e[0]+1.41,e[0]+10.61],[e[1]+1.41,e[1]+10.61],"k",linewidth=10.0)

def getRandTheta(dList,x,y):

    rAndThetaList=[]

    for i in dList:
        #the r part
        r=sqrt((i[0]-x)**2+(i[1]-y)**2)
    
        #the theta part
        side=i[2]
        #traslating the origin to the position of the detector
        xl=x-i[0]
        yl=y-i[1]
        if side=="lower":
            v=(1,0)
        elif side=="right":
            v=(0,1)
        elif side=="upper":
            v=(-1,0)
        elif side=="left":
            v=(0,-1)
        elif side=="llc":
            v=(1/sqrt(2),-1/sqrt(2))
        elif side=="lrc":
            v=(1/sqrt(2),1/sqrt(2))
        elif side=="urc":
            v=(-1/sqrt(2),1/sqrt(2))
        elif side=="ulc":
            v=(-1/sqrt(2),-1/sqrt(2))
        else:
            print "Entered else "+ side
            v=(0,0)
    
        cosTheta=(xl*v[0]+yl*v[1])/sqrt(xl**2+yl**2)
        
        theta=acos(cosTheta)
        rAndThetaList.append((r,theta))

    return rAndThetaList

#Making a list with the generated signals

def getSignals(dList,x,y):

    rTList=getRandTheta(dList,x,y)
    #print rTList
    sList=[]

    A=10
    gamma=2.0

    for e in rTList:
        r=e[0]
        theta=e[1]
        S=A*sin(theta)/r**gamma
        sList.append(S)


    return sList



#For proving its correct

def getPointBack(dList,x,y):
    rTList=getRandTheta(dList,x,y)
   

    counter=0
    
    
    for e in rTList:
        
        side=dList[counter][2]
       
        r=e[0]
        theta=e[1]

        xl=r*cos(theta)
        yl=r*sin(theta)

        xd=dList[counter][0]
        yd=dList[counter][1]
        
        if side=="lower":
            x=xl+xd
            y=yl+yd

        elif side=="upper":
            x=-xl+xd
            y=-yl+yd
        
        elif side=="left":
            x=yl-xd
            y=-xl+yd
        
        elif side=="right":
            x=-yl+xd
            y=xl+yd

        else:
            print "Its a corner" + dList[counter][2]




        print x,y
        counter+=1

#comparing with ANGER

def getAngerPos(dList,x,y):
    S=getSignals(dList,x,y)

    sSum=0


    for e in S:
        sSum+=e

    print sSum

    xP,yP=0,0


    counter=0
    for e in S:
        xP+=e*dList[counter][0]/sSum
        yP+=e*dList[counter][1]/sSum

        counter+=1

    print xP,yP


#Making the dictionary
def getLexicon(dList,platePolygon,N):

    lexicon={}
    xyPoints=[getRandomInPlate(platePolygon) for i in range(N)]
    
    for p in xyPoints:
        x,y=p
        maxListString=str(getMaxList(dList,x,y))

        if maxListString not in lexicon:
            lexicon[maxListString]=[]
        lexicon[maxListString].append(p)

    return lexicon

#Making mini dictionary
def getMiniLexicon(lexicon,argNo=3):

    miniLex={}

    for e in lexicon:
        ee=[int(u) for u in e[1:-1].split(",")]
        redListStr=str(ee[0:argNo])

        if redListStr not in miniLex:
            miniLex[redListStr]=[]

        miniLex[redListStr]+=lexicon[e]

    return miniLex


plotDetector(dList)

N=10000

lexicon=getLexicon(dList,platePolygon,N)

miniLexicon=getMiniLexicon(lexicon,5)

miniList=miniLexicon["[0, 8, 1, 15, 7]"]

plotList(miniList)

#plotMaxRegions(dList,platePolygon,N)

#maxRegions=getMaxRegions(dList,platePolygon,N)
#
#plotFirstSecondMax(maxRegions,4,5,"rx")
#
#plotFirstSecondMax(maxRegions,4,6,"bo")
#
#plotFirstSecondMax(maxRegions,4,3,"y^")
#
#plotFirstSecondMax(maxRegions,4,2,"c^")

#plotGeometry()

plt.show()




