import matplotlib.pyplot as plt
from shapely.geometry import Polygon,MultiPoint,Point
from shapely.ops import cascaded_union
import random
from math import *


#define los puntos iniciales, el numero de puntos, el valor A, la posicion del detector
def myCurve(x0=0,y0=0,np=50,A=5,side="lower"):
    def getR(A,ang,gamma):
        return (A*sin(ang))**(1.0/gamma)
    points=[]
    
    dang=pi/np
    ang=0
    gamma=2.0
    
    
    #Para el barrido angular de los detectores de la parte inferior
    if side=="lower":
        for i in range(np):
            r=getR(A,ang,gamma)
            points.append((x0+r*cos(ang),y0+r*sin(ang)))
            ang += dang


    #Para el barrido angular de los detectores de la parte izquierda
    elif side=="left":
        for i in range(np):
            r=getR(A,ang,gamma)
            points.append((x0+r*sin(ang),y0+r*cos(ang)))
            ang += dang


    #Para el barrido angular de los detectores de la parte derecha
    elif side=="right":
        for i in range(np):
            r=getR(A,ang,gamma)
            points.append((x0-r*sin(ang),y0+r*cos(ang)))
            ang += dang

    #Para el barrido angular de los detectores de la parte superior
    elif side=="upper":
        for i in range(np):
            r=getR(A,ang,gamma)
            points.append((x0+r*cos(ang),y0-r*sin(ang)))
            ang += dang

    #Para el barrido angular de los detectores de esquina superior izquierda
    elif side=="ulc":
        for i in range(np):
            r=getR(A,ang,gamma)
            points.append((x0+r*cos(ang-pi/4.),y0-r*sin(ang-pi/4.)))
            ang += dang

    #Para el barrido angular de los detectores de esquina superior derecha
    elif side=="urc":
        for i in range(np):
            r=getR(A,ang,gamma)
            points.append((x0+r*cos(ang+pi/4.),y0-r*sin(ang+pi/4.)))
            ang += dang

    #Para el barrido angular de los detectores de esquina inferior izquierda
    elif side=="llc":
        for i in range(np):
            r=getR(A,ang,gamma)
            points.append((x0+r*cos(ang-pi/4.),y0-r*sin(ang-pi/4.)))
            ang += dang

    #Para el barrido angular de los detectores de esquina inferior derecha
    elif side=="lrc":
        for i in range(np):
            r=getR(A,ang,gamma)
            points.append((x0+r*cos(ang+pi/4.),y0-r*sin(ang+pi/4.)))
            ang += dang

    return points



dList=[(79.35,0,"lower","on"),#this
              (49.35,0,"lower","on"),
              (34.35,0,"lower","on"),
              (2.5,2.5,"llc","off"),#corner
              (0,35,"left","on"),
              (2.5,67.5,"ulc","off"),#corner
              (34.35,70,"upper","on"),
              (49.35,70,"upper","on"),
              (79.35,70,"upper","on"),#this
              (109.35,70,"upper","on"),
              (124.35,70,"upper","on"),
              (156.35,67.5,"urc","off"),#corner
              (158.7,35,"right","on"),
              (156.2,2.5,"lrc","off"),#corner
              (124.35,0,"lower","on"),
              (109.35,0,"lower","on")]

def plotGeometry():
    
    A=2000
    polypoints=100

    dPoints=[] #detector points
    convexPoints=[]
    
    polygons=[]
    polyCoords=[]
    px,py=[],[]
    pltPoly=[]

    for e in dList:
        dPoints.append(myCurve(e[0],e[1],polypoints,A,e[2]))
        polygons.append(0)
        polyCoords.append(0)
        px.append(0)
        py.append(0)
        pltPoly.append(0)
        convexPoints.append(0)

    for i in range(len(dList)):
        if dList[i][3]=="off":
            continue
        
        convexPoints[i]=list(MultiPoint(dPoints[i]).convex_hull.exterior.coords)

        polygons[i]=Polygon(convexPoints[i])

        polyCoords[i]=list(polygons[i].exterior.coords)

        px[i]=[x[0] for x in polyCoords[i]]
        py[i]=[y[1] for y in polyCoords[i]]

        plt.plot(px[i],py[i],'bs')

        pltPoly[i]=plt.Polygon(polyCoords[i], fc="b")
        plt.gca().add_patch(pltPoly[i])


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

    plt.show()


plotGeometry()
#myCurve(5)