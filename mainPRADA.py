from PRADAfunctions import *
from realDataStuff import *



#plotPetals()
#plt.show()

#plotDetector(plateList,dList)
#N=100000

# saveLexicon(dList,platePolygon,N,file_name="lexicon100kNEc.pkl")
lexicon=openLexicon("lexicon1MNEc.pkl")
#
#miniLexicon=getMiniLexicon(lexicon,1)
#miniList=miniLexicon["[0]"]
#print(len(miniList))
#plotPoints(miniList)
#plt.show()

#plotMiniLexiconPoints(lexicon,1,"None")
#plt.show()

#polMiniLexList=openPolMiniLexList("polMiniLexListParallel1MNEcSpecialAlpha1.pkl")
#plotPolMiniLexList(polMiniLexList,3)
#plt.show()

############################################
#POLYGONS BOUNDARIES
#lexicon=openLexicon("lexicon1MNEc.pkl")
#
#miniLexicon=getMiniLexicon(lexicon,2)
#
#X=np.array(miniLexicon["[4]"])
#
#Y=miniLexicon["[4]"]
#
#pointList=convList2Point(Y)
#
#concave_hull, edge_points = alpha_shape(pointList,alpha=0.1)
#fig=plt.figure()
#_ = plot_polygon(concave_hull,fig)
#x = [p.coords.xy[0] for p in pointList]
#y = [p.coords.xy[1] for p in pointList]
#plt.plot(x,y,'o', color='b')
#
#point_collection = MultiPoint(list(pointList))
#point_collection.envelope
#
#convex_hull_polygon = point_collection.convex_hull
#_ = plot_polygon(concave_hull,fig)
#_ = plot_polygon(point_collection.envelope,fig)
#testVar = plt.plot(x,y,'o', color='b')

#print ("type(testVar) = ",type(testVar))
#plt.show()

############################################################
#CLUSTERS

#miniLexicon=getMiniLexicon(lexicon,2)
#Y=miniLexicon["[0, 8]"]
#
#clusterList=getClusterList(Y)
#
#x=[e[0] for e in clusterList[0]]
#y=[e[1] for e in clusterList[0]]
#
#xp=[e[0] for e in clusterList[1]]
#yp=[e[1] for e in clusterList[1]]
#
#polygonList1=getCluster2Polygon(clusterList,alpha=0.5)
#
#fig=plt.figure()
#for poly in polygonList1:
#    fig=plot_polygon(poly,fig)
#fig=plot_polygon(polygonList1[0],fig)
#fig=plot_polygon(polygonList1[1],fig)
#print ("type = ",type(polygonList1[0]))
#plt.gca().add_patch(plt.Polygon(polygonList1[0],fc="g"))

#############################################################

#Y=miniLexicon["[6, 7]"]
#
#clusterList=getClusterList(Y)
#
#polygonList2=getCluster2Polygon(clusterList,alpha=0.5)
#
#fig=plt.figure()
#
#for poly in polygonList2:
#    fig=plot_polygon(poly,fig)
#
#for dectComb in miniLexicon:
#    print (dectComb)
#    Y=miniLexicon[dectComb]
#    clusterList=getClusterList(Y)
#    if clusterList==False:
#        continue
#    polygonList=getCluster2Polygon(clusterList,alpha=0.05)
#    break
#    fc=getRandColor()
#    for poly in polygonList:
#        #This should be implemented in getCLuster2Polygon
#        #Maybe even from alpha_shape
#        fig=plot_polygon(poly,fig,fc)
#
#
#plot_polygon(polygonList[0],fig)

###############################################################


#
#for i in range(16):
#    plotPolyMiniLex(polMiniLexList[i])
#
#for poly in certPolMiniLexDict100kNEc:
#    fig=plot_polygon(poly,fig)
#plot_polygon(polMiniLexDict[4],fig)

#for poly in certPolMiniLexDict100kNEc:
#    fig=plot_polygon(poly,fig)

#plt.plot(x,y,"o", color="b")
#plt.plot(xp,yp,"o", color="r")

# plt.show()

#polyPartMiniLex=getPolyPartMiniLex(1,"polMiniLexList10k.pkl")
#fig=plt.figure()
#fig=plotPolyMiniLex(polyPartMiniLex,fig)
#plot_polygon(polyPartMiniLex["[8, 7, 9]"],fig)





