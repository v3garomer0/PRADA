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

from random import gauss

import plotly.plotly as py

from scipy import optimize

def myCurve(x0=0,y0=0,np=50,A=10,S=1,side="lower"):
    """Gets the petals points for each PMT"""
    def getR(A,S,ang,gamma):
        """Gets the distance R, according to formula"""
#        #print("S = ",S)
#        def f(r):
#            #print "f(r)=",f(r)
#            return S-A*sin(ang)/r**gamma
#        R=optimize.bisect(f,10e-7,100)
#        print "ang,R",ang,R
#        return R
        return (A/S*sin(ang))**(1.0/gamma)
    points=[]
    
    dang=pi/np
    ang=0.1
    gamma=1.3
    lamb=210
    np-=5

    #Para el barrido angular de los detectores de la parte inferior
    if side=="lower":
        print "side =",side
        for i in range(np):
            r=getR(A,S,ang,gamma)
            points.append((x0+r*cos(ang),y0+r*sin(ang)))
            ang += dang

    #Para el barrido angular de los detectores de la parte izquierda
    elif side=="left":
        print "side =",side
        for i in range(np):
            r=getR(A,S,ang,gamma)
            points.append((x0+r*sin(ang),y0+r*cos(ang)))
            ang += dang

    #Para el barrido angular de los detectores de la parte derecha
    elif side=="right":
        print "side =",side
        for i in range(np):
            r=getR(A,S,ang,gamma)
            points.append((x0-r*sin(ang),y0+r*cos(ang)))
            ang += dang

    #Para el barrido angular de los detectores de la parte superior
    elif side=="upper":
        print "side =",side
        for i in range(np):
            r=getR(A,S,ang,gamma)
            points.append((x0+r*cos(ang),y0-r*sin(ang)))
            ang += dang

    #Para el barrido angular de los detectores de esquina superior izquierda
    elif side=="ulc":
        print "side =",side
        for i in range(np):
            r=getR(A,S,ang,gamma)
            points.append((x0+r*cos(ang-pi/4.),y0-r*sin(ang-pi/4.)))
            ang += dang

    #Para el barrido angular de los detectores de esquina superior derecha
    elif side=="urc":
        print "side =",side
        for i in range(np):
            r=getR(A,S,ang,gamma)
            points.append((x0+r*cos(ang+pi/4.),y0-r*sin(ang+pi/4.)))
            ang += dang

    #Para el barrido angular de los detectores de esquina inferior izquierda
    elif side=="llc":
        print "side =",side
        for i in range(np):
            r=getR(A,S,ang,gamma)
            points.append((x0+r*cos(ang-pi/4.),y0+r*sin(ang-pi/4.)))
            ang += dang

    #Para el barrido angular de los detectores de esquina inferior derecha
    elif side=="lrc":
        print "side =",side
        for i in range(np):
            r=getR(A,S,ang,gamma)
            points.append((x0+r*cos(ang+pi/4.),y0+r*sin(ang+pi/4.)))
            ang += dang

    return points

#list of the photomultiplier tubes
dList=[(79.35,0,"lower","on"),
       (49.35,0,"lower","off"),
       (34.35,0,"lower","off"),
       (2.5,2.5,"llc","off"),#corner
       (0,35,"left","off"),
       (2.5,67.5,"ulc","off"),#corner
       (34.35,70,"upper","off"),
       (49.35,70,"upper","off"),
       (79.35,70,"upper","off"),
       (109.35,70,"upper","off"),
       (124.35,70,"upper","off"),
       (156.35,67.5,"urc","off"),#corner
       (158.7,35,"right","off"),
       (156.2,2.5,"lrc","off"),#corner
       (124.35,0,"lower","off"),
       (109.35,0,"lower","off")]

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

def getRandTheta(dList,x,y):
    """Gets a list with r and theta, given a point"""
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
def getSignals(dList,x,y,petalSignals=False):
    """Gets the signals list, given a point"""
    rTList=getRandTheta(dList,x,y)
    sList=[]
    
    A=10
    gamma=1.3
    lamb=210
    thetaC=pi/6
    
    for e in rTList:
        r=e[0]
        theta=e[1]
        if petalSignals:
            S=(A*sin(theta))/r**gamma
        elif thetaC>theta>0 or pi>theta>pi-thetaC:
            S=0
        else:
            S=A*exp(-r/lamb)*sin(pi*(theta-thetaC)/(pi-2*thetaC))/r**gamma
            
        sList.append(S)

    return sList

#checking if the point is inside the plate
def checkIfInPlate(platePolygon,x,y):
    """Tests if a point is within MONDEs plate"""
    point=Point(x,y)

    return platePolygon.contains(point)

#giving a random point within the plate
def getRandomInPlate(platePolygon):
    """Gets a point within MONDEs plate"""
    x=random.uniform(0,158.7)
    y=random.uniform(0,70)

    while not checkIfInPlate(platePolygon,x,y):
        x=random.uniform(0,158.7)
        y=random.uniform(0,70)

    return [x,y]

#generating a list, ordering the max signals
def getMaxList(dList,x,y):
    """Gets a list in which the PMTs are sorted from the one with the
    maximum signal to the one with the minimum signal

    """
    sList=getSignals(dList,x,y)
    
    return sorted(range(len(sList)), key=lambda k: sList[k], reverse=True)

#ordering detectors from real data
def getMaxRealList(realSList):
    """Gets a list in which the PMTs are sorted from the one with the
    maximum signal to the one with the minimum signal, from a list of
    signal data

    """
    maxRealList=sorted(range(len(realSList)), key=lambda k: realSList[k], reverse=True)
    return maxRealList

#given a detector order and a combination, returns a combination with less detectors
def getRealStringList4Dict(dectOrder,maxRealList):
    """Makes a short string list of the MaxRealList for the dictionary"""
    shortList=maxRealList[dectOrder:]
    stringList4Dict=str(shortList)
    return stringList4Dict

#for plotting petals
def plotPetals(x=80,y=35):
    """Makes the plot of the polygon petals for each detector, gets the area
    of intersection and obtains the centroid of the intersected
    polygon

    """
    A=10
    polypoints=100
    #plotDetector(plateList,dList)

    dPoints=[] #detector points
    convexPoints=[]
    
    polygons=[]
    polyCoords=[]
    px,py=[],[]
    pltPoly=[]
    
    counter=0

    sList=getSignals(dList,x,y,True)
    
    for e in dList:
        S=sList[counter]
        print (counter,S)
        if S==0:
            dPoints.append(False)
            continue
        #S=S-0.1*S
        dPoints.append(myCurve(e[0],e[1],polypoints,A,S,e[2]))
        polygons.append(0)
        polyCoords.append(0)
        px.append(0)
        py.append(0)
        pltPoly.append(0)
        convexPoints.append(0)
    
        counter+=1

    for i in range(len(dList)):
        if dPoints[i]==False:
            convexPoints[i]=False
        else:
            convexPoints[i]=list(MultiPoint(dPoints[i]).convex_hull.exterior.coords)

        #generating the polygons properly
        polygons[i]=Polygon(convexPoints[i])
        polyCoords[i]=list(polygons[i].exterior.coords)

        if dList[i][-1]=="off":
            continue

        px[i]=[x[0] for x in polyCoords[i]]
        py[i]=[y[1] for y in polyCoords[i]]

        plt.plot(px[i],py[i],'r')

#        pltPoly[i]=plt.Polygon(polyCoords[i], fc="b")
#        plt.gca().add_patch(pltPoly[i])

#        interPol=polygons[i]
#    counter = 0
#    for p in polygons:
#        if dList[counter][3]=="on":
##            print ("dList val", dList[counter][3])
#            interPol=interPol.intersection(p)
#        counter+=1
##        centroidCoord=list(interPol.centroid.coords)[0]
#        interPolList=list(interPol.exterior.coords)
#
#    centroidCoord=list(interPol.centroid.coords)[0]
#    print ("Centroid = ",centroidCoord)
#
#    AreaPol=interPol.area
#    
#    print ("Area Intersection= ",AreaPol)

#    pIx=[x[0] for x in interPolList]
#    pIy=[y[1] for y in interPolList]

#    plt.plot(pIx,pIy,'go')
#
#    pltPolyIntersect = plt.Polygon(interPolList, fc='g')

#    plt.gca().add_patch(pltPolyIntersect)

#Plotting MONDEs plate
def plotDetector(plateList,dList):
    """Plots MONDE's Plate in 2D with the respectives PMTs"""
    fig=plt.figure()
    x=[e[0]for e in plateList]
    y=[e[1] for e in plateList]
    plt.xlabel('Posicion [cm]')
    plt.ylabel('Posicion [cm]')
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
    return fig

#MONDE 3D
def plotDetector3D(plateList,dList):
    """Plots MONDE's Plate in 3D with the respectives PMTs"""
    fig=plt.figure()
    ax=fig.gca(projection='3d')

    x=[e[0]for e in plateList]
    y=[e[1] for e in plateList]
    z=0
    ax.plot(x,y,z,"y",linewidth=2.0)
    ax.legend()
    ax.set_xlabel('Posicion[cm]')
    ax.set_ylabel('Posicion[cm]')
    ax.set_zlabel('Posicion[cm]')
    
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

#For proving its correct
def getPointBack(dList,x,y):
    """Given a point, it checks if for every r and theta generated, it
    returns the same given point

    """
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
def getAngerPosFromPos(dList,x,y):
    """Get the points generated with an ANGER algorithm"""
    S=getSignals(dList,x,y)
    sSum=0
    for e in S:
        sSum+=e
    xP,yP=0,0
    counter=0
    for e in S:
        xP+=e*dList[counter][0]/sSum
        yP+=e*dList[counter][1]/sSum
        counter+=1
    print (xP,yP)

#Making the dictionary
def getLexicon(dList,platePolygon,N):
    """Generates a dictionary using N points, in which each entry is a
    combination of detectors and it contains the points for that
    combination

    """
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
    """Creates a mini dictionary from the big dictionary, using only the
    entries with the desired number of PMTs

    """
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
    """Generates and saves a file with the dictionary using N points"""
    lexicon=getLexicon(dList,platePolygon,N)
    
    fileObject=open(file_name,'wb')
    pickle.dump(lexicon,fileObject,2)
    fileObject.close()

#opening file with the dictionary
def openLexicon(file_name="lexicon.pkl"):
    """Opens the file with the dictionary"""
    fileObject=open(file_name,'rb')

    lex=pickle.load(fileObject)
    
    fileObject.close()

    return lex

################################################################################
#Polygon boundaries

#making list of points from dictionary
# Y would be a miniLexicon
def convList2Point(Y):
    """Converts the points in a miniLexicon to a point object"""
    pointsList=[]
    for e in Y:
        pointsList.append(Point(e))
        print (e)
    return pointsList

def plot_polygon(polygon,fig,fc='#999999'):
    """Plots a polygon object with a random color"""
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
    """Gets a random color number"""
    r = lambda: random.randint(0,255)
    fc='#%02X%02X%02X' % (r(),r(),r())
    return fc

#sample function, needs 2 be changed
def getColor4HeatMap(counts,maxCount):
    """Gets the color number for the heatmap"""
    colorValR=int(255*sin(pi/2*counts/maxCount))
    colorValG=int(255*sin(pi*counts/maxCount))
    colorValB=int(255*cos(pi/2*counts/maxCount))
    fc='#%02X%02X%02X' % (colorValR,colorValG,colorValB)
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
        return MultiPoint(list(pointList)).convex_hull,False
    def add_edge(edges, edge_points, coords, i, j):
        """Add a line between the i-th and j-th points, if not in the list
            already

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
    """Makes a list with the formed clusters from a list of points"""
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
        l=int(l)
        if l==-1:
            continue

        clusterList[l].append(p)

    return clusterList

def getCluster2Polygon(clusterList,alpha=0.1):
    """Makes a list with the polygons obtained from the clusters"""
    polygonList=[]

    for e in clusterList:
        pointsObjt=convList2Point(e)
        concave_hull, edge_points=alpha_shape(pointsObjt,alpha=alpha)

        if type(concave_hull) != shapely.geometry.polygon.Polygon:
            print ("Entered if in getCluster2Polygon")
            print (type(concave_hull))
            continue
        polygonList.append(concave_hull)

    return polygonList

###################################################################

def getPolygons4MiniLex(miniLex,alpha=0.1):
    """Makes a dictionary of polygons up to a certain number of combination
    of detectors, using a miniLexicon

    """
    polMiniLex={}

    for e in miniLex:
        lexPoints=miniLex[e]
        clusterList=getClusterList(lexPoints)
        if clusterList==False:
            print ("Entered false cond", len(e), e)
            continue
        clusterList=getCluster2Polygon(clusterList,alpha=alpha)
        if clusterList != []:
            polMiniLex[e]=clusterList

    return polMiniLex

def getPolMiniLexList(lexicon,alpha=0.1):
    """Makes a list of polygons for each combination of PMTs, using a
    lexicon

    """
    polMiniLexList=[]
    
    for i in range(16):
        #print ("i + 1 = ",i + 1)
        tempMiniLex=getMiniLexicon(lexicon,argNo=i+1)
        tempPolMiniLex=getPolygons4MiniLex(tempMiniLex,alpha)
        polMiniLexList.append(tempPolMiniLex)
    
    return polMiniLexList

#########################################################################

#saving file with the list of polygons
def savePolMiniLexList(lexicon,file_name="polMiniLexList.pkl"):
    """Creates and saves a file with the list of polygons for each
    combination of PMTs, using a lexicon"""
    polMiniLexList=getPolMiniLexList(lexicon,alpha=0.1)
    
    fileObject=open(file_name,"wb")
    pickle.dump(polMiniLexList,fileObject,2)
    fileObject.close()

#opening file with the list of polygons
def openPolMiniLexList(file_name="polMiniLexList.pkl"):
    """Opens the file with the list of polygons for each combination of PMTs"""
    fileObject=open(file_name,"rb")
    polMiniLexList=pickle.load(fileObject)
    
    fileObject.close()
    
    return polMiniLexList

#############################################################################
#getting the dictionary element of the list
def getPolyPartMiniLex(dectAmount=3,polMiniLexList="polMiniLexList.pkl"):
    """Gets a part of the polMiniLexList which takes into account only a
    certain number of detectors

    """
    bigList=openPolMiniLexList(polMiniLexList)
    polyPartMiniLex=bigList[dectAmount-1]
    return polyPartMiniLex

def plotPolyMiniLex(polyPartMiniLex):
    """Plots the polygons from the list polyPartMiniLex which only considers
    a certain number of detectors

    """
    fig=plotDetector(plateList,dList)
    #dont forget to call plt.show()  after the function
    for dectComb in polyPartMiniLex:
        fc=getRandColor()
        print (dectComb)
        for poly in polyPartMiniLex[dectComb]:
            fig=plot_polygon(poly,fig,fc)
    return fig

def plotPolMiniLexList(polMiniLexList,combOrder=1,detectOfInterest="none"):
    """Plots just the polygons of polMiniLexList which are of interest and
    with the number of detectors desired

    """
    fc=getRandColor()
    for detectOrder in range(len(polMiniLexList)):
        #fc=getRandColor()
        if detectOrder==combOrder:
            break
        fig=plotDetector(plateList,dList)
        for detectComb in polMiniLexList[detectOrder]:
            intList=string2List(detectComb)
            if detectOfInterest!="none":
                if detectOfInterest != intList[0]:
                    continue
            #fc=getRandColor()
            subPolList=polMiniLexList[detectOrder][detectComb]
            for poly in subPolList:
                fig=plot_polygon(poly,fig,fc)

def string2List(stringDetectComb):
    """Converts a string to an integer"""
    intList=[int(u) for u in stringDetectComb[1:-1].split(",")]
    return intList

def plotPoints(pointsList,fig="None"): #points list would be a miniLex
    """Given a list of coordinates, plots the points of the list"""
    fc=getRandColor()
    if fig == "None":
        fig=plotDetector(plateList,dList)
    x=[e[0] for e in pointsList]
    y=[e[1] for e in pointsList]
    plt.plot(x,y,"o",color=fc)
     

def plotMiniLexiconPoints(lexicon,combOrder=1,detectOfInterest="None"):
    """Plots the points of a miniLexicon"""
    fig=plotDetector(plateList,dList)
    miniLexicon=getMiniLexicon(lexicon,combOrder)
    for e in miniLexicon:
        intList=string2List(e)
        if detectOfInterest!="None":
            if detectOfInterest != intList[0]:
                continue
        subPointsList=miniLexicon[e]
        fc=getRandColor()
        plotPoints(subPointsList,fig)

################################################################################
#Function for certain number of detectors
def getCertPolMiniLex(lexicon,orderNo,alpha=0.1):
    """Gets a dictionary with polygons of a particular alpha from the
    lexicon with a certain number of detectors

    """
    tempMiniLex=getMiniLexicon(lexicon,orderNo)
    certPolMiniLexDict=getPolygons4MiniLex(tempMiniLex,alpha=0.1)
    
    return certPolMiniLexDict


#saving dictionary with a certain number of detectors
def saveCertainPolMiniLexDict(orderNo,lexicon,file_name="CertPolMiniLexDict.pkl"):
    """Creates and saves a file with the dictionary with polygons of a
    certain alpha and for a particular number of detectors

    """
    certPolMiniLexDict=getCertPolMiniLex(lexicon,orderNo,alpha=0.1)
    
    fileObject=open(file_name,"wb")
    pickle.dump(certPolMiniLexDict,fileObject,2)
    fileObject.close()


def openCertainPolMiniLexDict(file_name="CertPolMiniLexDict.pkl"):
    """Opens the file with the dictionary with polygons of a certain alpha
    and for a particular number of detectors

    """
    fileObject=open(file_name,"rb")
    certPolMiniLexDict=pickle.load(fileObject)
    
    fileObject.close()
    
    return certPolMiniLexDict

def plotCertPolyMiniLex(certPolMiniLexDict):
    """Plots the polygons of the dictionary of polygons with a certain alpha
    and for a particular number of detectors

    """
    fig=plt.figure()
    plotDetector(plateList,dList)
    
    for detectOrder in certPolMiniLexDict:
        for poly in certPolMiniLexDict[detectOrder]:
            fc=getRandColor()
            fig=plot_polygon(poly,fig,fc)

    return fig

################################################################################
#plotting in 3D

def plot2DPolyIn3D(ax,poly,fc,zVal=0.0):
    """Plots in 3D the 2D polygons"""
    polyListCoords=list(poly.exterior.coords)
    poly=PolyCollection([polyListCoords],facecolors=[fc],closed=False)
    zs=[zVal]
    ax.add_collection3d(poly,zs=zs,zdir='z')
    return ax

def plotPolyMiniLex3D(polyPartMiniLex):
    """Plots the polygons from the list polyPartMiniLex which only considers
    a certain number of detectors in 3D

    """
#    fig=plt.figure()
    ax=plotDetector3D(plateList,dList)
    ax.set_zlim(0, 10)
    #dont forget to call plt.show()  after the function
    for dectComb in polyPartMiniLex:
        # print (dectComb)
        fc=getRandColor()
        zVal=random.randint(0,10)
        for poly in polyPartMiniLex[dectComb]:
            ax=plot2DPolyIn3D(ax,poly,fc,zVal)

##########################################################
#HEATMAP
def getCountDict(detectCombList):
    """Given a list of detector combinations, it returns a dictionary with
    the counts for each combination

    """
    countDict={}
    for e in detectCombList:
        sE = str(e)

        if sE not in countDict:
            countDict[sE]=1
        else:
            countDict[sE]+=1

    return countDict

def getCombNotInMapList(countDict, polyPartMiniLex):
    """Gets a list with the combinations that dont appear in the dictionary
    countDict

    """
    combNotInMapList=[e for e in countDict if e not in polyPartMiniLex]
    return combNotInMapList

def getMaxCount(countDict):
    """Gets the maximum value of counts in the dictionary countDict"""
    maxCount=max([countDict[e] for e in countDict])
    return maxCount

def plotCountDictPolygons(countDict, polyPartMiniLex):
    """Plots a heatmap in accordance with the number of counts each
    combination of PMTs have

    """
    fig=plotDetector(plateList,dList)
    #dont forget to call plt.show()  after the function
    maxCount=getMaxCount(countDict)
    for dectComb in polyPartMiniLex:
        if dectComb not in countDict:
            counts=0
        else:
            counts=countDict[dectComb]
        # fc=getRandColor()#Change this to the heatmap funct stuff...
        fc=getColor4HeatMap(counts,maxCount)
        for poly in polyPartMiniLex[dectComb]:
            fig=plot_polygon(poly,fig,fc)
    return fig

#######################################################################################
#Real Data
def getDiscriminateDetectList(shortDataInList,threshold=50,dectNo=3):
    """Gets a list with the real data but only with values over the threshold"""
    discriminateDetectList=[]
    for e in shortDataInList:
        miniDDList=getMiniDiscList(e,threshold,dectNo)
        if miniDDList!=[]:
            discriminateDetectList.append(miniDDList)
    return discriminateDetectList

def getMiniDiscList(listRow,threshold=50,dectNo=3):
    """Gets a list with the real data up to the desired number of detectors"""
    miniDiscList=[i for i in listRow if i>=threshold]
    if len(miniDiscList)>=dectNo:
        detectList=getMaxListFromData(listRow)
        return detectList[:dectNo]
    return[]

def getMaxListFromData(dataList):
    """Gets a list with the detectors sorted from the one with the maximum
    value to the one with the minimum value

    """
    return sorted(range(len(dataList)), key=lambda k: dataList[k], reverse=True)

##################################################################
#Z Part
def getSignalsZ(dList,x,y,z=0):
    """Gives a list with the calculated signals for each PMT, considering
    the z coordinate

    """
    rTList=getRandTheta(dList,x,y)
    sZList=[]
    rThetaPhiList=[]
    A=10
    gamma=1.3
    lamb=210
    thetaC=pi/6
    
    for e in rTList:
        theta=e[1]
        phiZ=pi/2-atan(z/e[0])
        riZ=sqrt(e[0]**2+z**2)
        rThetaPhiList.append([riZ,theta,phiZ])
    
    for e in rThetaPhiList:
        if thetaC>e[1]>0 or pi>e[1]>pi-thetaC or thetaC>e[2]>0 or pi>e[2]>pi-thetaC:
            S=0
        else:
            S=A*exp(-e[0]/lamb)*sin(pi*(e[2]-thetaC)/(pi-2*thetaC))*sin(pi*(e[1]-thetaC)/(pi-2*thetaC))/e[0]**gamma
        sZList.append(S)
    
    return sZList

#def getMaxZList(dList,x,y,z):
#    sZList=getSignalsZ(dList,x,y,z)
#    
#    return sorted(range(len(sZList)), key=lambda k: sZList[k], reverse=True)

def getMaxZList(sZList):
    """Gets a list in which the PMTs are sorted from the one with the
    maximum signal to the one with the minimum signal, considering the z
    coordinate

    """
    maxZList=getMaxRealList(sZList)
    return maxZList

def getWeirdZPoints(dList,z,noPoints,detOrder):
    """Gives a list with the points which, for a given point, the order of
    detectors does not match with the order of the ones considering the
    z coordinate

    """
    weirdZPointsList=[]
    
    for i in range(noPoints):
        x,y=getRandomInPlate(platePolygon)
        normalSignals=getSignals(dList,x,y)
        zSignals=getSignalsZ(dList,x,y,z)
        normalOrder=getMaxList(dList,x,y)
        zOrder=getMaxZList(zSignals)
        if normalOrder[:detOrder]!=zOrder[:detOrder]:
            weirdZPointsList.append([x,y])

    return weirdZPointsList

#################################################################
#Gaussian Simulation Part

def getRandomGauss(centroidx,centroidy,sigma):
    """Gives a gaussian distribution from a random centroid point"""
    if not checkIfInPlate(platePolygon,centroidx,centroidy):
        print ("Centroid not in plate")
        return

    x=gauss(centroidx,sigma)
    y=gauss(centroidy,sigma)
    
    while not checkIfInPlate(platePolygon,x,y):
        x=gauss(centroidx,sigma)
        y=gauss(centroidy,sigma)
    

    return x,y

def createGaussSim(centroidx,centroidy,sigma,N,fileName="gaussTest.pkl"):
    listOfSignals=[]
    for e in range(N):
        x,y=getRandomGauss(centroidx,centroidy,sigma)
        print (x,y)
        signals=getSignals(dList,x,y)
        listOfSignals.append(signals)
        print (signals)
    fileObject=open(fileName,'wb')
    pickle.dump(listOfSignals,fileObject,2)
    fileObject.close()

###############################################################################
# Number of Polygons, number of polygons with a certain area, number of regions
def countPolygons(polMiniLexListPkl="polMiniLexListParallel1MNEcSpecialAlpha1.pkl"):
    polMiniLexList=openPolMiniLexList(polMiniLexListPkl)
    polNumList=[]
    for myDictionary in polMiniLexList:
        polyCount=0
        for combination in myDictionary:
            polyCount+=len(myDictionary[combination])
        polNumList.append(polyCount)
    return polNumList

def plotCountPolygons(polMiniLexListPkl="polMiniLexListParallel1MNEcSpecialAlpha1.pkl"):
    polNumList=countPolygons(polMiniLexListPkl)
    x=[i+1 for i in range(16)]
    y=[e for e in polNumList]
    plt.subplot(111)
    plt.plot(x,y,'k',x,y,'ro')
    plt.title("Number of Polygons Generated")
    plt.xlabel("Combination Order")
    plt.ylabel("Number of Polygons")
    plt.grid(True)
    for a,b in zip(x, y):
        plt.text(a, b, str(b))
    plt.show()

def areaPolygons(polMiniLexListPkl="polMiniLexListParallel1MNEcSpecialAlpha1.pkl",order=1):
    polMiniLexList=openPolMiniLexList(polMiniLexListPkl)
    polAreaList=[]
    myDictionary=polMiniLexList[order-1]
    for combination in myDictionary:
        for myPoly in myDictionary[combination]:
            areaPol=myPoly.area
            polAreaList.append(areaPol)
    return polAreaList

def histAreaList(polMiniLexListPkl="polMiniLexListParallel1MNEcSpecialAlpha1.pkl",order=1):
    polALst=areaPolygons(polMiniLexListPkl,order)
    plt.hist(polALst)
    plt.title("Area Histogram")
    plt.xlabel("Area")
    plt.ylabel("Number of Polygons")
    plt.grid(True)
    fig = plt.gcf()
    plt.show()

def countRegions(polMiniLexListPkl="polMiniLexListParallel1MNEcSpecialAlpha1.pkl"):
    polMiniLexList=openPolMiniLexList(polMiniLexListPkl)
    regNumList=[]
    regCount=0
    for myDictionary in polMiniLexList:
        regCount+=len(myDictionary)
        regNumList.append(regCount)
    return regNumList

def plotCountRegions(polMiniLexListPkl="polMiniLexListParallel1MNEcSpecialAlpha1.pkl"):
    regNumList=countRegions(polMiniLexListPkl)
    x=[i+1 for i in range(16)]
    y=[e for e in regNumList]
    plt.plot(x,y,'k',x,y,'ro')
    plt.title("Number of Regions Obtained")
    plt.xlabel("Combination Order")
    plt.ylabel("Number of Regions")
    plt.grid(True)
    for a,b in zip(x, y):
        plt.text(a, b, str(b))
    plt.show()

def getMaxArea(polMiniLexListPkl="polMiniLexListParallel1MNEcSpecialAlpha1.pkl"):
    maxAreaList=[]
    for i in range(16):
        areaList=areaPolygons(polMiniLexListPkl,i+1)
        maxAreaList.append(max(areaList))
    return maxAreaList

def getMinArea(polMiniLexListPkl="polMiniLexListParallel1MNEcSpecialAlpha1.pkl"):
    minAreaList=[]
    for i in range(16):
        areaList=areaPolygons(polMiniLexListPkl,i+1)
        minAreaList.append(min(areaList))
    return minAreaList

def plotMaxArea(polMiniLexListPkl="polMiniLexListParallel1MNEcSpecialAlpha1.pkl"):
    maxAreaList=getMaxArea(polMiniLexListPkl)
    x=[i+1 for i in range(16)]
    y=[e for e in maxAreaList]
    plt.plot(x,y,'k',x,y,'ro')
    plt.title("Maximum Area for Each Combination Order")
    plt.xlabel("Combination Order")
    plt.ylabel("Maximum Area")
    plt.grid(True)
    for a,b in zip(x, y):
        plt.text(a, b, str(round(b,2)))
    plt.show()

def plotMinArea(polMiniLexListPkl="polMiniLexListParallel1MNEcSpecialAlpha1.pkl"):
    minAreaList=getMinArea(polMiniLexListPkl)
    x=[i+1 for i in range(16)]
    y=[e for e in minAreaList]
    plt.plot(x,y,'k',x,y,'ro')
    plt.title("Minimum Area for Each Combination Order")
    plt.xlabel("Combination Order")
    plt.ylabel("Minimum Area")
    plt.grid(True)
    for a,b in zip(x, y):
        plt.text(a, b, str(round(b,3)))
    plt.show()

########################################################################################
#Getting the position (x,y)

def getAngerWeightAreaPos(polygonList):
    xList=[]
    yList=[]
    AList=[]
    ATot=0
    for poly in polygonList:
        centCoords=list(poly.centroid.coords)
        x,y=centCoords[0]
        A=poly.area
        xList.append(x)
        yList.append(y)
        AList.append(A)
        ATot+=A
    xProm=0
    for xi,Ai in zip(xList,AList):
        xProm+=xi*Ai
    xProm/=ATot
    yProm=0
    for yi,Ai in zip(yList,AList):
        yProm+=yi*Ai
    yProm/=ATot
    return [xProm,yProm]

def getCentroidLex(certPolLexList):
    certPolCentroidLex={}
    for certPolList in certPolLexList:
        if certPolList==[]:
            continue
        c=getAngerWeightAreaPos(certPolLexList[certPolList])
        certPolCentroidLex[certPolList]=c
    return certPolCentroidLex

def getCentroidLexList(polLexList):
    centroidPointsLexList=[]
    for myDict in polLexList:
        lex=getCentroidLex(myDict)
        centroidPointsLexList.append(lex)
    return centroidPointsLexList

def pickleCentroidLexList(polLexList="polMiniLexListParallel1MNEcSpecialAlpha1.pkl",file_name="centroidPointsLexList.pkl"):
    polLexList=openPolMiniLexList(polLexList)
    centroidPointsLexList=getCentroidLexList(polLexList)

    fileObject=open(file_name,"wb")
    pickle.dump(centroidPointsLexList,fileObject,2)
    fileObject.close()

def openCentroidLexList(file_name="centroidPointsLexList.pkl"):
    """Opens the file with the list of polygons centroids for each combination of PMTs"""
    fileObject=open(file_name,"rb")
    centroidPointsLexList=pickle.load(fileObject)
    fileObject.close()
    return centroidPointsLexList

###############################################################################
#comparing points of anger and prada

def getAngerPosFromSignal(sList,dList):
    sSum=0
    for signals in sList:
        sSum+=signals
    xP,yP=0,0
    counter=0
    for signals in sList:
        xP+=signals*dList[counter][0]/sSum
        yP+=signals*dList[counter][1]/sSum
        counter+=1
    return (xP,yP)

def getPRADAPosfromSignals(sList,order=3,centroidPointsLexList="centroidPointsLexList.pkl"):
    centroidPointsLexList=openCentroidLexList(centroidPointsLexList)
    centroidPointsLexListOrder=centroidPointsLexList[order-1]
    maxCentroidList=sorted(range(len(sList)), key=lambda k: sList[k], reverse=True)
    maxCentroidListStr=str(maxCentroidList[0:order])
    if maxCentroidListStr in centroidPointsLexListOrder:
        return centroidPointsLexListOrder[maxCentroidListStr]

def printAngerAndPRADAComparison(sList,dList,order=3,centroidPointsLexList="centroidPointsLexList.pkl"):
    xA,yA=getAngerPosFromSignal(sList,dList)
    xP,yP=getPRADAPosfromSignals(sList,order,centroidPointsLexList)

    return ((xA,yA),(xP,yP))
