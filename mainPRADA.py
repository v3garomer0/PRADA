import matplotlib.pyplot as plt
from shapely.geometry import Polygon,MultiPoint,Point
from shapely.ops import cascaded_union
import random
from math import *


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

    sList=getSignals(dList,80,35)
    
    
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
        if dList[i][3]=="off":
            continue
    
        if dList[i][2]=="lower":
            ldetector=Polygon([(e[0]-2.23,e[1]),(e[0]+2.23,e[1]),(e[0]+2.23,e[1]-20.8),(e[0]-2.23,e[1]-20.8)])
    
        convexPoints[i]=list(MultiPoint(dPoints[i]).convex_hull.exterior.coords)

        polygons[i]=Polygon(convexPoints[i])

        polyCoords[i]=list(polygons[i].exterior.coords)

        px[i]=[x[0] for x in polyCoords[i]]
        py[i]=[y[1] for y in polyCoords[i]]

        plt.plot(px[i],py[i],'r')

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


#    plt.xlim(-200,200)
#    plt.ylim(-200,200)


    plt.show()



#myCurve(5)


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

plotGeometry()

