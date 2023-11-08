from matplotlib import pyplot as plt
import numpy as np

class utils:
    def __init__(self,function,history,ranges=[10,10],stages=50) -> None:
        self.x_range=ranges[0]
        self.y_range=ranges[1]
        self.stages=stages
        self.function=function
        self.history=history
    def draw_cotours(self):
        xrange = np.arange(-self.x_range, self.x_range, 0.01)
        yrange = np.arange(-self.y_range, self.y_range, 0.01)
        c=np.arange(-self.stages,self.stages,5)
        X, Y = np.meshgrid(xrange,yrange)
        plt.contour(X,Y,self.function([X,Y]),c)
        plt.savefig('./resualts/contour.jpg')
        plt.show()
    def draw_optim_path(self):
        x=self.history['x']
        x_range=max(x[:,0])+2
        y_range=max(x[:,1])+2
        stages=abs(max(self.history['values']))+2
        xrange = np.arange(-x_range, x_range, 0.01)
        yrange = np.arange(-y_range, y_range, 0.01)
        c=np.arange(-stages,stages,0.01*stages)
        X, Y = np.meshgrid(xrange,yrange)
        plt.contour(X,Y,self.function([X,Y]),c)
        plt.plot(self.history['x'][:,0],self.history['x'][:,1])
        plt.savefig('./resualts/path.jpg')
        plt.show()
    def plot_hist(self):
        pass
    