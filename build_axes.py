import matplotlib.pyplot as plt
import matplotlib.gridspec as gs


class Axes:

    def __init__(self, x, y, title, grid_axis='y', nbins=10):
        self.x = x
        self.y = y
        self.title = title
        self.grid_axis = grid_axis
        # self.tick_params = tick_params
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

    def plot(self, figsize=(12, 4), dpi=500):

        plt.style.use('ggplot')

        fig = plt.figure(figsize=figsize, dpi=dpi)
        grid = gs.GridSpec(nrows=1, ncols=2)

        ax1 = plt.subplot(grid[0, 0])
        ax1.barh(self.x, self.y)
        ax1.set_title(self.title)
        ax1.grid(axis=self.grid_axis)
        ax1.tick_params(left=False, bottom=False)

        if self.nbins < 10:
            ax1.locator_params(axis='x', nbins=self.nbins)

        ax2 = plt.subplot(grid[0, 1])
        ax2.set_facecolor('w')
        ax2.tick_params(axis='both', colors='w')

        plt.tight_layout(h_pad=2, w_pad=2)

        return ax1, ax2


class Barh:

    def __init__(self, x, y, title, nbins=10, grid_axis='y'):
        self.x = x
        self.y = y
        self.title = title
        self.nbins = nbins
        self.grid_axis = grid_axis
        
    def subplot(self, grid, row, column):
        """Build an axes with a horizontal bar chart inside a grid.

        Parameters:
            grid (Gridspec object):
            row (Integer):
            column (Integer):

        Returns:
            axes: 
        """

        ax = plt.subplot(grid[row, column])
        ax.barh(self.x, self.y)
        ax.set_title(self.title)
        ax.grid(axis=self.grid_axis)
        ax.tick_params(left=False, bottom=False)
        
        if self.nbins != 10:
            ax.locator_params(axis='x', nbins=self.nbins)

        return ax


class Hist:

    def __init__(self, x, title, bins=10, grid_axis='x'):
        self.x = x
        self.title = title
        self.bins = bins
        self.grid_axis = grid_axis
        
    def subplot(self, grid, row, column):
        """Build an axes with a histogram inside a grid.

        Parameters:
            grid (Gridspec object):
            row (Integer):
            column (Integer):

        Returns:
            axes: 
        """

        ax = plt.subplot(grid[row, column])
        ax.hist(self.x, bins=self.bins)
        ax.set_title(self.title)
        ax.grid(axis=self.grid_axis)
        ax.tick_params(left=False, bottom=False)
        
        return ax