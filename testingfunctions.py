from PRADAfunctions import *
from matplotlib.patches import Circle, PathPatch
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
import math


#
#plotDetector(dList) #plot MONDE
#
#plotGeometry(40,20) #plot regions of the PMT

#def plot_polygon(polygon):
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    margin = .3
#    x_min, y_min, x_max, y_max = polygon.bounds
#    ax.set_xlim([x_min-margin, x_max+margin])
#    ax.set_ylim([y_min-margin, y_max+margin])
#    patch = PolygonPatch(polygon, fc='#999999',
#                         ec='#000000', fill=True,
#                         zorder=-1)
#    ax.add_patch(patch)
#    return fig

#x = np.linspace(0, 2*np.pi, 400)
#y = np.sin(x**2)
#
## Just a figure and one subplot
#f, ax = plt.subplots()
#ax.plot(x, y)
#ax.set_title('Simple plot')
#
#def plot_polygon3D(polygon,fig,fc='#999999'):
#    #fig = plt.figure()
#    ax = fig.add_subplot(111)
#    margin = .3
    #    x_min, y_min, x_max, y_max = polygon.bounds
    #    ax.set_xlim([x_min-margin, x_max+margin])
    #    ax.set_ylim([y_min-margin, y_max+margin])
#    patch = PolygonPatch(polygon, fc=fc,
#                         ec='#000000', fill=True,
#                         zorder=-1)
#    ax.add_patch(patch)
#    return fig
#
#def plotDetector3D(plateList,dList):
#    fig=plt.figure()
#    ax=fig.gca(projection='3d')
#
#    x=[e[0]for e in plateList]
#    y=[e[1] for e in plateList]
#    z=0
#    ax.plot(x,y,z,"y",linewidth=2.0)
#    ax.legend()
#    
#    #Generating each photomultiplier tube
#    for e in dList:
#        if e[3]=="off":
#            continue
#        
#        if e[2]=="lower":
#            ax.plot([e[0],e[0]],[e[1]-2,e[1]-17],0,"k",linewidth=10.0)
#            ax.legend()
#        if e[2]=="upper":
#            ax.plot([e[0],e[0]],[e[1]+2,e[1]+17],0,"k",linewidth=10.0)
#            ax.legend()
#        if e[2]=="right":
#            ax.plot([e[0]+2,e[0]+17],[e[1],e[1]],0,"k",linewidth=10.0)
#            ax.legend()
#        if e[2]=="left":
#            ax.plot([e[0]-2,e[0]-17],[e[1],e[1]],0,"k",linewidth=10.0)
#            ax.legend()
#        if e[2]=="llc":
#            ax.plot([e[0]-1.41,e[0]-10.61],[e[1]-1.41,e[1]-10.61],0,"k",linewidth=10.0)
#            ax.legend()
#        if e[2]=="ulc":
#            ax.plot([e[0]-1.41,e[0]-10.61],[e[1]+1.41,e[1]+10.61],0,"k",linewidth=10.0)
#            ax.legend()
#        if e[2]=="lrc":
#            ax.plot([e[0]+1.41,e[0]+10.61],[e[1]-1.41,e[1]-10.61],0,"k",linewidth=10.0)
#            ax.legend()
#        if e[2]=="urc":
#            ax.plot([e[0]+1.41,e[0]+10.61],[e[1]+1.41,e[1]+10.61],0,"k",linewidth=10.0)
#            ax.legend()
#
#    return ax
#
#def plot2DPolyIn3D(ax,poly,fc):
#    polyListCoords=list(poly.exterior.coords)
#    poly=PolyCollection([polyListCoords],facecolors=[fc],closed=False)
#    
#    zs=[0.0]
#    
#    ax.add_collection3d(poly,zs=zs,zdir='z')
#
#    return ax
#
#def plotPolyMiniLex3D(polyPartMiniLex):
#    fig=plt.figure()
#    ax=plotDetector3D(plateList,dList)
#    #dont forget to call plt.show() after the function
#    for dectComb in polyPartMiniLex:
#        print (dectComb)
#        fc=getRandColor()
#        for poly in polyPartMiniLex[dectComb]:
#            ax=plot2DPolyIn3D(ax,poly,fc)


## Two subplots, unpack the output array immediately
#f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#ax1.plot(x, y)
#ax1.set_title('Sharing Y axis')
#ax2.scatter(x, y)
#
## Four polar axes
#plt.subplots(2, 2, subplot_kw=dict(polar=True))
#
## Share a X axis with each column of subplots
#plt.subplots(2, 2, sharex='col')
#
## Share a Y axis with each row of subplots
#plt.subplots(2, 2, sharey='row')
#
## Share a X and Y axis with all subplots
#plt.subplots(2, 2, sharex='all', sharey='all')
## same as
#plt.subplots(2, 2, sharex=True, sharey=True)

#plt.show()



def tSig(r):
    lamb=210
    gamma=1.3
    theta=pi/10
    A=2
    S=0.5
#    thetac=pi/6
    return S-A*exp(-r/lamb)*sin(theta)/r**gamma

def bisection(f, a, b, TOL=0.001, NMAX=100):
	"""
	Takes a function f, start values [a,b], tolerance value(optional) TOL and
	max number of iterations(optional) NMAX and returns the root of the equation
	using the bisection method.
	"""
	n=1
	while n<=NMAX:
		c = (a+b)/2.0
		print "a=%s\tb=%s\tc=%s\tf(c)=%s"%(a,b,c,f(c))
		if f(c)==0 or (b-a)/2.0 < TOL:
			return c
		else:
			n = n+1
			if f(c)*f(a) > 0:
				a=c
			else:
				b=c
	return False
