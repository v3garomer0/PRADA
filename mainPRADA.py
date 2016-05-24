from PRADAfunctions import *

fig=pl.figure()
plotDetector(dList)
#
N=10000
#
#saveLexicon(dList,platePolygon,N,file_name="lexicon10k.pkl")
# lexicon=openLexiconFile("lexicon10k.pkl")
# #
# miniLexicon=getMiniLexicon(lexicon,2)

# miniList=miniLexicon["[0, 8]"]
# plotList(miniList)



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
lexicon=openLexiconFile("lexicon100k.pkl")
#
#miniLexicon=getMiniLexicon(lexicon,1)
#
#X=np.array(miniLexicon["[4]"])
#
#Y=miniLexicon["[4]"]
#
#pointList=convList2Point(Y)
#
#concave_hull, edge_points = alpha_shape(pointList,alpha=0.01)
##fig=pl.figure()
#_ = plot_polygon(concave_hull,fig)
#
#x = [p.coords.xy[0] for p in pointList]
#y = [p.coords.xy[1] for p in pointList]
##pl.figure(figsize=(10,10))
##pl.plot(x,y,'o', color='b')
#
#point_collection = MultiPoint(list(pointList))
#point_collection.envelope
#
###convex_hull_polygon = point_collection.convex_hull
###_ = plot_polygon(convex_hull_polygon)
###_ = plot_polygon(point_collection.envelope)
#testVar = pl.plot(x,y,'o', color='b')
#print ("type(testVar) = ",type(testVar))
##pl.show()

##################################################
#CLUSTERS

miniLexicon=getMiniLexicon(lexicon,3)
#Y=miniLexicon["[0, 8]"]
#
#clusterList=getClusterList(Y)
#
#x=[e[0] for e in clusterList[0]]
#y=[e[1] for e in clusterList[0]]

#xp=[e[0] for e in clusterList[1]]
#yp=[e[1] for e in clusterList[1]]

#polygonList1=getCluster2Polygon(clusterList,alpha=0.5)

#fig=pl.figure()
#for poly in polygonList1:
#    fig=plot_polygon(poly,fig)
#fig=plot_polygon(polygonList1[0],fig)
#fig=plot_polygon(polygonList1[1],fig)
#print ("type = ",type(polygonList1[0]))
#pl.gca().add_patch(pl.Polygon(polygonList1[0],fc="g"))

#Y=miniLexicon["[6, 7]"]
#
#clusterList=getClusterList(Y)
#
#polygonList2=getCluster2Polygon(clusterList,alpha=0.5)
#
#for poly in polygonList2:
#    fig=plot_polygon(poly,fig)

for dectComb in miniLexicon:
    print (dectComb)
    Y=miniLexicon[dectComb]
    clusterList=getClusterList(Y)
    if clusterList==False:
        continue
    polygonList=getCluster2Polygon(clusterList,alpha=0.05)
    print (polygonList)
    if polygonList==False:
        continue
    fc=getRandColor()
    for poly in polygonList:
        #This should be implemented in getCLuster2Polygon
        #Maybe even from alpha_shape
        print ("type(poly) = ",type(poly))
        if type(poly) != shapely.geometry.polygon.Polygon:
            print("Entered condition")
            continue
        fig=plot_polygon(poly,fig,fc)

#Y=miniLexicon["[9, 10]"]
#
#clusterList=getClusterList(Y)
#
#polygonList3=getCluster2Polygon(clusterList,alpha=0.5)
#
#for poly in polygonList3:
#    fig=plot_polygon(poly,fig)

#pl.plot(x,y,"o", color="b")
#pl.plot(xp,yp,"o", color="r")

pl.show()

#polyPartMiniLex=getPolyPartMiniLex(1,"polMiniLexDict10k.pkl")
#fig=pl.figure()
#fig=plotPolyMiniLex(polyPartMiniLex,fig)
#plot_polygon(polyPartMiniLex["[8, 7, 9]"],fig)

#pl.show()
