from PRADAfunctions import *

plotDetector(dList)
#
#N=10000
#
#lexicon=openLexiconFile("lexicon100k.pkl")
#
#miniLexicon=getMiniLexicon(lexicon,2)
#
#miniList=miniLexicon["[0, 8]"]
#plotList(miniList)



#miniList=miniLexicon["[3, 5]"]
#plotList(miniList)
#
#miniList=miniLexicon["[15, 9]"]
#plotList(miniList)

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

############################################
#POLYGONS BOUNDARIES
lexicon=openLexiconFile("lexicon10k.pkl")

miniLexicon=getMiniLexicon(lexicon,1)

X=np.array(miniLexicon["[4]"])

Y=miniLexicon["[4]"]

pointList=convList2Point(Y)

concave_hull, edge_points = alpha_shape(pointList,alpha=0.01)
_ = plot_polygon(concave_hull)

x = [p.coords.xy[0] for p in pointList]
y = [p.coords.xy[1] for p in pointList]
#pl.figure(figsize=(10,10))
#pl.plot(x,y,'o', color='b')

point_collection = geometry.MultiPoint(list(pointList))
point_collection.envelope

##convex_hull_polygon = point_collection.convex_hull
##_ = plot_polygon(convex_hull_polygon)
##_ = plot_polygon(point_collection.envelope)
_ = pl.plot(x,y,'o', color='b')

pl.show()

##################################################
#CLUSTERS

miniLexicon=getMiniLexicon(lexicon,2)
Y=miniLexicon["[3, 2]"]

clusterList=getClusterList(Y)

x=[e[0] for e in clusterList[0]]
y=[e[1] for e in clusterList[0]]

#xp=[e[0] for e in clusterList[1]]
#yp=[e[1] for e in clusterList[1]]

polygonList1=getCluster2Polygon(clusterList,alpha=0.5)
plot_polygon(polygonList1[0])

plt.plot(x,y,"o", color="b")
#plt.plot(xp,yp,"o", color="r")



plt.show()

