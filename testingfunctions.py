from PRADAfunctions import *

plotDetector(dList) #plot MONDE

plotGeometry(40,20) #plot regions of the PMT

#def plot_polygon(polygon):
#    fig = pl.figure()
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


pl.show()