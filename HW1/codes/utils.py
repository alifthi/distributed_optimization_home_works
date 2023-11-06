from matplotlib import pyplot as plt
import numpy as np

class utils:
    def __init__(self,function,ranges=[10,10],stages=50) -> None:
        self.x_range=ranges[0]
        self.y_range=ranges[1]
        self.stages=stages
        self.function=function
    def draw_cotours(self):
        xrange = np.arange(-self.x_range, self.x_range, 0.01)
        yrange = np.arange(-self.y_range, self.y_range, 0.01)
        c=np.arange(0,self.stages,2)
        X, Y = np.meshgrid(xrange,yrange)
        plt.contour(X,Y,self.function([X,Y]),c)
        plt.show()
    def draw_optim_path(self):
        pass
    def plot_hist(self):
        pass
    