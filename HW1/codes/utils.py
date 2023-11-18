from matplotlib import pyplot as plt
import numpy as np

class utils:
    def __init__(self,function,history,name,ranges=[10,10],stages=50) -> None:
        self.x_range=ranges[0]
        self.y_range=ranges[1]
        self.stages=stages
        self.function=function
        self.history=history
        self.name=name
    def draw_cotours(self):
        if self.history['x'].shape[1] >2:
            return
        xrange = np.arange(-self.x_range, self.x_range, 0.01)
        yrange = np.arange(-self.y_range, self.y_range, 0.01)
        c=np.arange(-self.stages,self.stages,5)
        X, Y = np.meshgrid(xrange,yrange)
        plt.contour(X,Y,self.function(np.array([X,Y])),c)
        plt.savefig(f'./resualts/{self.name}_contour.jpg')
        plt.show()
    def draw_optim_path(self):
        if self.history['x'].shape[1] >2:
            return
        x=self.history['x']
        x_range=max(x[:,0])+2
        y_range=max(x[:,1])+2
        if x_range>10 or y_range>10:
            y_range=10
            x_range=10
        stages=abs(max(self.history['values']))+2
        if stages>50:
            stages=50
        xrange = np.arange(-x_range, x_range, 0.01)
        yrange = np.arange(-y_range, y_range, 0.01)
        c=np.arange(-stages,stages,0.01*stages)
        X, Y = np.meshgrid(xrange,yrange)
        plt.contour(X,Y,self.function([X,Y]),c)
        plt.plot(self.history['x'][:,0],self.history['x'][:,1])
        plt.savefig(f'./resualts/{self.name}_path.jpg')
        plt.show()
    def plot_hist(self):
        plt.plot(self.history['iterations'],self.history['values'])
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.savefig(f'./resualts/{self.name}_history.jpg')
        plt.show()
    