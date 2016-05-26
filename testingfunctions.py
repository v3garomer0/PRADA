from PRADAfunctions import *
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

def plotDetectorNew(plateList):
    x=[e[0]for e in plateList]
    y=[e[1] for e in plateList]
    plt.plot(x,y,"y",linewidth=2.0)
#    plt.plot([5,153.7],[0,0],"y",linewidth=2.0)
#    plt.plot([153.7,158.7],[0,5],"y",linewidth=2.0)
#    plt.plot([158.7,158.7],[5,65],"y",linewidth=2.0)
#    plt.plot([158.7,153.7],[65,70],"y",linewidth=2.0)
#    plt.plot([153.7,5],[70,70],"y",linewidth=2.0)
#    plt.plot([5,0],[70,65],"y",linewidth=2.0)
#    plt.plot([0,0],[65,5],"y",linewidth=2.0)
#    plt.plot([0,5],[5,0],"y",linewidth=2.0)

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

plt.show()