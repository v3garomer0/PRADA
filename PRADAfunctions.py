from shapely.geometry import Polygon,MultiPoint,Point,MultiLineString
from shapely.ops import cascaded_union, polygonize
import random
from math import *
import pickle

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

from descartes import PolygonPatch
from scipy.spatial import Delaunay

import shapely

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.patches import Circle, PathPatch
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter

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
       (0,65),
        (0,5)]

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

    return [x,y]

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

     

def plotGeometry(x=80,y=55):
    A=10
    polypoints=100

    dPoints=[] #detector points
    convexPoints=[]
    
    polygons=[]
    polyCoords=[]
    px,py=[],[]
    pltPoly=[]
    
    counter=0

    sList=getSignals(dList,x,y)
    
    for e in dList:
        S=sList[counter]
        print (counter,S)

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

        if dList[i][-1]=="off":
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
#            print ("dList val", dList[counter][3])
            interPol=interPol.intersection(p)
        counter+=1
#        centroidCoord=list(interPol.centroid.coords)[0]
        interPolList=list(interPol.exterior.coords)

    centroidCoord=list(interPol.centroid.coords)[0]
    print ("Centroid = ",centroidCoord)

    AreaPol=interPol.area
    
    print ("Area Intersection= ",AreaPol)

    pIx=[x[0] for x in interPolList]
    pIy=[y[1] for y in interPolList]

#    plt.plot(pIx,pIy,'go')

    pltPolyIntersect = plt.Polygon(interPolList, fc='g')
    
    plt.gca().add_patch(pltPolyIntersect)

     

#Plotting MONDEs plate
def plotDetector(plateList,dList):
    x=[e[0]for e in plateList]
    y=[e[1] for e in plateList]
    plt.plot(x,y,"y",linewidth=2.0)

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

#MONDE 3D
def plotDetector3D(plateList,dList):
    fig=plt.figure()
    ax=fig.gca(projection='3d')

    x=[e[0]for e in plateList]
    y=[e[1] for e in plateList]
    z=0
    ax.plot(x,y,z,"y",linewidth=2.0)
    ax.legend()
    
    #Generating each photomultiplier tube
    for e in dList:
        if e[3]=="off":
            continue
        
        if e[2]=="lower":
            ax.plot([e[0],e[0]],[e[1]-2,e[1]-17],0,"k",linewidth=10.0)
            ax.legend()
        if e[2]=="upper":
            ax.plot([e[0],e[0]],[e[1]+2,e[1]+17],0,"k",linewidth=10.0)
            ax.legend()
        if e[2]=="right":
            ax.plot([e[0]+2,e[0]+17],[e[1],e[1]],0,"k",linewidth=10.0)
            ax.legend()
        if e[2]=="left":
            ax.plot([e[0]-2,e[0]-17],[e[1],e[1]],0,"k",linewidth=10.0)
            ax.legend()
        if e[2]=="llc":
            ax.plot([e[0]-1.41,e[0]-10.61],[e[1]-1.41,e[1]-10.61],0,"k",linewidth=10.0)
            ax.legend()
        if e[2]=="ulc":
            ax.plot([e[0]-1.41,e[0]-10.61],[e[1]+1.41,e[1]+10.61],0,"k",linewidth=10.0)
            ax.legend()
        if e[2]=="lrc":
            ax.plot([e[0]+1.41,e[0]+10.61],[e[1]-1.41,e[1]-10.61],0,"k",linewidth=10.0)
            ax.legend()
        if e[2]=="urc":
            ax.plot([e[0]+1.41,e[0]+10.61],[e[1]+1.41,e[1]+10.61],0,"k",linewidth=10.0)
            ax.legend()
    return ax


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
            print ("Entered else "+ side)
            v=(0,0)
    
        cosTheta=(xl*v[0]+yl*v[1])/sqrt(xl**2+yl**2)
        theta=acos(cosTheta)
        rAndThetaList.append((r,theta))

    return rAndThetaList

#Making a list with the generated signals

def getSignals(dList,x,y):
    rTList=getRandTheta(dList,x,y)
    #print (rTList)
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
            print ("Its a corner" + dList[counter][2])

        print (x,y)
        counter+=1

#comparing with ANGER
def getAngerPos(dList,x,y):
    S=getSignals(dList,x,y)

    sSum=0

    for e in S:
        sSum+=e

    print (sSum)

    xP,yP=0,0

    counter=0
    for e in S:
        xP+=e*dList[counter][0]/sSum
        yP+=e*dList[counter][1]/sSum

        counter+=1

    print (xP,yP)

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

#saving file with the dictionary
def saveLexicon(dList,platePolygon,N,file_name="lexicon.pkl"):
    lexicon=getLexicon(dList,platePolygon,N)
    
    fileObject=open(file_name,'wb')
    pickle.dump(lexicon,fileObject)
    fileObject.close()

#opening file with the dictionary
def openLexiconFile(file_name="lexicon.pkl"):
    fileObject=open(file_name,'rb')

    lex=pickle.load(fileObject)
    
    fileObject.close()

    return lex

################################################################################
#Testing polygon boundaries
def convList2Point(Y):
    pointsList=[]
    for e in Y:
        pointsList.append(Point(e))
    
    return pointsList

def plot_polygon(polygon,fig,fc='#999999'):
    #fig = plt.figure()
    ax = fig.add_subplot(111)
    margin = .3
#    x_min, y_min, x_max, y_max = polygon.bounds
#    ax.set_xlim([x_min-margin, x_max+margin])
#    ax.set_ylim([y_min-margin, y_max+margin])
    patch = PolygonPatch(polygon, fc=fc,
                         ec='#000000', fill=True,
                         zorder=-1)
    ax.add_patch(patch)
    return fig

def getRandColor():
    r = lambda: random.randint(0,255)
    fc='#%02X%02X%02X' % (r(),r(),r())
    return fc
    
def alpha_shape(pointList, alpha):
    """
        Compute the alpha shape (concave hull) of a set
        of points.
        @param points: Iterable container of points.
        @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
        """
    if len(pointList) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return MultiPoint(list(pointList)).convex_hull
    def add_edge(edges, edge_points, coords, i, j):
        """
            Add a line between the i-th and j-th points,
            if not in the list already
            """
        if (i, j) in edges or (j, i) in edges:
                # already added
                return
        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])
    coords = np.array([point.coords[0]
                       for point in pointList])
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
        # Semiperimeter of triangle
        s = (a + b + c)/2.0
        # Area of triangle by Heron's formula
        area = sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)
        # Here's the radius filter.
        #print (circum_r)
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
    m = MultiLineString(edge_points)
    triangles = list(polygonize(m))

    #print ("edge_points=", edge_points)

    return cascaded_union(triangles), edge_points

###################################################################
#CLUSTER SECTION

def getClusterList(aList):
    X=np.array(aList)
    
    db = DBSCAN(eps=2.5, min_samples=5).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    if n_clusters_==0:
        return False

    clusterList=[[] for i in range(n_clusters_)]

    for p,l in zip(X,labels):
        if l==-1:
            continue

        clusterList[l].append(p)

    return clusterList

def getCluster2Polygon(clusterList,alpha=0.01):
    polygonList=[]

    for e in clusterList:
        pointsObjt=convList2Point(e)
        concave_hull, edge_points=alpha_shape(pointsObjt,alpha=alpha)
        if type(concave_hull) != shapely.geometry.polygon.Polygon:
            print ("Entered condition")
            continue
        polygonList.append(concave_hull)

    return polygonList

###################################################################

def getPolygons4MiniLex(miniLex):
    polMiniLex={}

    for e in miniLex:
        lexPoints=miniLex[e]
        clusterList=getClusterList(lexPoints)
        if clusterList==False:
            print ("Entered false cond", len(e), e)
            continue
        clusterList=getCluster2Polygon(clusterList,alpha=0.5)
        if clusterList != []:
            polMiniLex[e]=clusterList

    return polMiniLex

def getPolMiniLex(lexicon):
    polMiniLexDict={}

    for i in range(16):
        print ("i + 1 = ",i + 1)
        tempMiniLex=getMiniLexicon(lexicon,argNo=i+1)
        tempPolMiniLex=getPolygons4MiniLex(tempMiniLex)
        polMiniLexDict[i+1]=tempPolMiniLex

    return polMiniLexDict

#########################################################################

#saving file with the dictionary of polygons
def savePolMiniLexDict(lexicon,file_name="polMiniLexDict.pkl"):
    polMiniLexDict=getPolMiniLex(lexicon)
    
    fileObject=open(file_name,"wb")
    pickle.dump(polMiniLexDict,fileObject)
    fileObject.close()

#opening file with the dictionary of polygons
def openPolMiniLexDict(file_name="polMiniLexDict.pkl"):
    fileObject=open(file_name,"rb")
    polMiniLexDict=pickle.load(fileObject)
    
    fileObject.close()
    
    return polMiniLexDict

#############################################################################

def getPolyPartMiniLex(dectAmount=3,polMiniLexDict="polMiniLexDict.pkl"):
    bigdict=openPolMiniLexDict(polMiniLexDict)
    polyPartMiniLex=bigdict[dectAmount]
    return polyPartMiniLex

def plotPolyMiniLex(polyPartMiniLex):
    fig=plt.figure()
    plotDetector(plateList,dList)
    #dont forget to call plt.show()  after the function
    for dectComb in polyPartMiniLex:
        fc=getRandColor()
        for poly in polyPartMiniLex[dectComb]:
            fig=plot_polygon(poly,fig,fc)
    return fig

def plotPoints(pointsList,fc): #points list would be a miniLex
    x=[e[0] for e in pointsList]
    y=[e[1] for e in pointsList]
    plt.plot(x,y,"o",color=fc)
     

def plotAllMiniLexiconPoints(miniLexicon):
    for e in miniLexicon:
        fc=getRandColor()
        plotPoints(miniLexicon[e],fc)
     

################################################################################
#3D PART

def getPolyPartMiniLexCounts(polyPartMiniLex):
    polyPartMiniLexCounts={}
    for e in polyPartMiniLex:
        polyPartMiniLexCounts[e]=[polyPartMiniLex[e],0]
    return polyPartMiniLexCounts


def plot2DPolyIn3D(ax,poly,fc,zVal=0.0):
    polyListCoords=list(poly.exterior.coords)
    poly=PolyCollection([polyListCoords],facecolors=[fc],closed=False)
    zs=[zVal]
    ax.add_collection3d(poly,zs=zs,zdir='z')
    return ax

def plotPolyMiniLex3D(polyPartMiniLex):
#    fig=plt.figure()
    ax=plotDetector3D(plateList,dList)
    ax.set_zlim(0, 10)
    #dont forget to call plt.show()  after the function
    for dectComb in polyPartMiniLex:
        print (dectComb)
        fc=getRandColor()
        zVal=random.randint(0,10)
        for poly in polyPartMiniLex[dectComb]:
            ax=plot2DPolyIn3D(ax,poly,fc,zVal)

     
