import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib import ticker

position_legend = 0

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


class BarhStacked2:

    def __init__(self, x1, x2, labels, title, legend, color1='#E24A33', color2='#777777', perc=False):
        self.x1 = x1
        self.x2 = x2
        self.labels = labels
        self.title = title
        self.legend = legend
        self.color1 = color1  # default is red
        self.color2 = color2  # default is light grey
        self.perc = perc  # deafult is to format the labels of the x axis as percentages

    def subplot(self, grid, row, column):
        """Build an axes with a horizontal bar chart inside a grid.

        Parameters:
            grid (Gridspec object):
            row (Integer):
            column (Integer):

        Returns:
            axes: 
        """

        def put_legend(rects, text, color):
            """Attach the legend text above and in the middle of the first rects bar.
            """

            left, right = ax.get_xlim()
            delta = right - left
            
            # plot the legends in the middle of the bars only if they have enough size
            if rects[-1].get_width() > delta*0.1 and rects[-1].get_width() < delta*0.9:    
                ax.annotate(text,
                            xy=(rects[-1].get_x() + rects[-1].get_width()/2, rects[-1].get_y() + rects[-1].get_height()),
                            xytext=(0, 5),  # 5 points vertical offset (padding)
                            textcoords="offset points",
                            ha='center', va='bottom',
                            color=color)

            # otherwise, plot the legends centralized in the axes
            else:
                global position_legend
                position_legend += delta/3
                ax.annotate(text,
                            xy=(position_legend, rects[-1].get_y() + rects[-1].get_height()),
                            xytext=(0, 5),  # 5 points vertical offset (padding)
                            textcoords="offset points",
                            ha='center', va='bottom',
                            color=color)

        n_classes = len(self.x1)
        space = 1/(4 + (n_classes - 1)*3)  # 1/4, 1/7, 1/10...

        # 1: the size of the y axis
        # (4 + (n_classes - 1)*3): the number of spaces to divide the y axis by.

        # The bar height is 2 spaces and between them is 1 space.
        # For example, if there are 2 classes in the y axis, the number of spaces nedded is:
        # 1 (from x axis to first bar)
        # + 2 (first bar height)
        # + 1 (between bars)
        # + 2 (second bar height)
        # + 1 (from second bar to the top limit)
        # = 7

        index = []
        for i in range(n_classes):
            position_multiplier = 2 + i*3
            index.append(space*position_multiplier)  # [space*2, space*5, space*8...]

        height = space*2

        ax = plt.subplot(grid[row, column])

        rects1 = ax.barh(index, self.x1, height=height, tick_label=self.labels, color=self.color1)  # set the first bar series color
        rects2 = ax.barh(index, self.x2, height=height, left=self.x1, color=self.color2)  # set the second bar series color

        ax.set_title(self.title, color='#555555', pad=20) # set dark grey as x axis color
        
        put_legend(rects1, text=self.legend[0], color=self.color1)
        put_legend(rects2, text=self.legend[1], color=self.color2)
        
        ax.set_facecolor('w')  # set white as background color

        ax.tick_params(left=False, width=1)  # set ticks width to 1, to match the spine line width
        ax.spines['bottom'].set_color('#555555')  # set dark grey as x axis color
        ax.spines['bottom'].set_linewidth(1)  # set spine line width to 1, to match the ticks width

        if self.perc:
            ax.xaxis.set_major_formatter(ticker.PercentFormatter())  # format x axis labels as percentages

        ax.margins(x=0, y=0)
        ax.set_ylim(0, 1)

        global position_legend
        position_legend = 0

        return ax


    def plot(self, figsize=(12, 4), dpi=500):

        def put_legend(rects, text, color):

            ax1.annotate(text,
                        xy=(rects[-1].get_x() + rects[-1].get_width()/2, rects[-1].get_y() + rects[-1].get_height()),
                        xytext=(0, 5),  # 5 points vertical offset (padding)
                        textcoords="offset points",
                        ha='center', va='bottom',
                        color=color)

        fig = plt.figure(figsize=figsize, dpi=dpi)
        grid = gs.GridSpec(nrows=1, ncols=2)

        n_classes = len(self.x1)
        space = 1/(4 + (n_classes - 1)*3)  # 1/4, 1/7, 1/10...

        index = []
        for i in range(n_classes):
            position_multiplier = 2 + i*3
            index.append(space*position_multiplier)  # [space*2, space*5, space*8...]

        height = space*2

        ax1 = plt.subplot(grid[0, 0])

        rects1 = ax1.barh(index, self.x1, height=height, tick_label=self.labels, color=self.color1)  # set the first bar series color
        rects2 = ax1.barh(index, self.x2, height=height, left=self.x1, color=self.color2)  # set the second bar series color

        ax1.set_title(self.title, color='#555555', pad=20) # set dark grey as x axis color
        
        put_legend(rects1, text=self.legend[0], color=self.color1)
        put_legend(rects2, text=self.legend[1], color=self.color2)
        
        ax1.set_facecolor('w')  # set white as background color

        ax1.tick_params(left=False, width=1)  # set ticks width to 1, to match the spine line width
        ax1.spines['bottom'].set_color('#555555')  # set dark grey as x axis color
        ax1.spines['bottom'].set_linewidth(1)  # set spine line width to 1, to match the ticks width

        if self.perc:
            ax1.xaxis.set_major_formatter(ticker.PercentFormatter())  # format x axis labels as percentages

        ax1.margins(x=0, y=0)
        ax1.set_ylim(0, 1)

        ax2 = plt.subplot(grid[0, 1])
        ax2.set_facecolor('w')
        ax2.tick_params(axis='both', colors='w')

        plt.tight_layout(h_pad=4, w_pad=4)

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