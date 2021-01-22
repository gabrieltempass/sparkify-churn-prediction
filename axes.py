import matplotlib.pyplot as plt

class Axes:

    def __init__(self, x, y, title, grid_axis='y', nbins=10):
        self.x = x
        self.y = y
        self.title = title
        self.grid_axis = grid_axis
        self.nbins = nbins
        
    def subplot(self, grid, row, column):
        ax = plt.subplot(grid[row, column])
        ax.barh(self.x, self.y)
        ax.set_title(self.title)
        ax.grid(axis=self.grid_axis)
        ax.tick_params(left=False, bottom=False)
        
        if self.nbins < 10:
            ax.locator_params(axis='x', nbins=self.nbins)
        
        return ax
