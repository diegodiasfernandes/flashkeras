from flashkeras.utils.otherimports import *
from sklearn.metrics import mean_squared_error # type: ignore

def plot_line_graph(y_values: list, 
                    x_values: list = None,
                    graph_title='Line Graph',
                    x_label='X', 
                    y_label='Y',
                    x_rotation=90,
                    fig_size=(15, 6),
                    grid: bool = False
                    ):

    if x_values is None:
        x_values = np.arange(1, len(y_values) + 1)

    # Plot the original data points
    plt.figure(figsize=fig_size)
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', label=x_label)

    # Add titles and labels
    plt.title(graph_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Rotate x-axis labels for readability
    plt.xticks(x_values, rotation=x_rotation)

    # Set y-axis ticks to divide into 20 parts
    y_min, y_max = min(y_values), max(y_values)
    y_ticks = np.linspace(y_min, y_max, 20)
    plt.yticks(y_ticks)

    # Disable grid, add legend, and show plot
    plt.grid(grid)
    plt.legend()
    plt.show()

def plot_multi_line_graph(y_values: list[list], 
                          x_values: list = None,
                          graph_title='Line Graph',
                          x_label='X', 
                          y_label='Y',
                          labels: list = None, 
                          x_rotation=90,
                          fig_size=(15, 6),
                          grid: bool = False
                          ):
    # Define x_values as default if not provided
    if x_values is None:
        x_values = np.arange(1, len(y_values[0]) + 1)

    # Create figure for plotting
    plt.figure(figsize=fig_size)

    # If no labels provided, create default labels for each line
    if labels is None:
        labels = [f'Line {i+1}' for i in range(len(y_values))]

    # Plot each line with a different color
    for i, y in enumerate(y_values):
        plt.plot(x_values, y, marker='o', linestyle='-', label=labels[i])

    # Add titles and labels
    plt.title(graph_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Rotate x-axis labels for readability
    plt.xticks(x_values, rotation=x_rotation)

    # Set y-axis ticks to divide into 20 parts based on all y-values
    all_y_values = np.concatenate(y_values)
    y_min, y_max = min(all_y_values), max(all_y_values)
    y_ticks = np.linspace(y_min, y_max, 20)
    plt.yticks(y_ticks)

    # Show grid, add legend, and display the plot
    plt.grid(grid)
    plt.legend()
    plt.show()

def plot_line_graph_polynomial_fit_broken_y_axis(y_values: list, 
                                               x_values: list = None,
                                               max_degree=5, 
                                               focus_threshold=0.8,
                                               graph_title='Graph with Polynomial Regression & Y axis Cut',
                                               x_label='X', 
                                               y_label='Y',
                                               x_rotation=90,
                                               fig_size=(15, 6),
                                               grid: bool = False
                                               ) -> None:

    def find_y_limits(y_values) -> tuple[float, float]:
        highest_value = max(y_values)
        lower_limit = highest_value * focus_threshold
        lower_dif = highest_value
        lower_value = lower_limit
        for v in y_values:
            dif = v - lower_limit
            if dif < lower_dif and dif >= 0:
                lower_value = v
                lower_dif = dif
        
        lower_value = lower_value - (lower_value / 150)
        highest_value = highest_value + (highest_value / 150)

        return lower_value, highest_value

    def find_best_polynomial_degree(x, y, max_degree):
        best_degree = 1
        best_error = float('inf')

        for degree in range(1, max_degree + 1):
            coefficients = np.polyfit(x, y, degree)
            polynomial = np.poly1d(coefficients)
            y_pred = polynomial(x)
            error = mean_squared_error(y, y_pred)

            if error < best_error:
                best_error = error
                best_degree = degree

        return best_degree

    if x_values is None:
        x_values = np.arange(1, len(y_values) + 1)

    # Find the best polynomial degree
    best_degree = find_best_polynomial_degree(x_values, y_values, max_degree)

    # Calculate the best polynomial regression
    coefficients = np.polyfit(x_values, y_values, best_degree)
    polynomial = np.poly1d(coefficients)
    regression_y_values = polynomial(x_values)

    # Create the plot
    plt.figure(figsize=fig_size)
    
    # Plot the original data points
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', label=x_label)

    # Plot the best fit regression line
    plt.plot(x_values, regression_y_values, linewidth=3, linestyle='--', color='r', label='Regression')

    # Add titles and labels
    plt.title(graph_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Rotate x-axis labels for readability
    plt.xticks(x_values, rotation=x_rotation)

    # Set y-axis limits to show only values above focus_threshold
    min_val, max_val = find_y_limits(y_values)
    plt.ylim(min_val, max_val)

    # Add the legend and show the plot
    plt.grid(grid)
    plt.legend()
    plt.show()

def plot_line_graph_polynomial_fit(y_values: list, 
                                   x_values: list = None,
                                   max_degree=5, 
                                   graph_title='Graph with Polynomial Regression',
                                   x_label='X', 
                                   y_label='Y',
                                   x_rotation=90,
                                   fig_size=(15, 6),
                                   grid: bool = False
                                   ):
    
    def find_best_polynomial_degree(x, y, max_degree):
        best_degree = 1
        best_error = float('inf')

        for degree in range(1, max_degree + 1):
            coefficients = np.polyfit(x, y, degree)
            polynomial = np.poly1d(coefficients)
            y_pred = polynomial(x)
            error = mean_squared_error(y, y_pred)

            if error < best_error:
                best_error = error
                best_degree = degree

        return best_degree

    if x_values is None:
        x_values = np.arange(1, len(y_values) + 1)

    # Find the best polynomial degree
    best_degree = find_best_polynomial_degree(x_values, y_values, max_degree)

    # Calculate the best polynomial regression
    coefficients = np.polyfit(x_values, y_values, best_degree)
    polynomial = np.poly1d(coefficients)
    regression_y_values = polynomial(x_values)

    # Plot the original data points
    plt.figure(figsize=fig_size)
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', label=x_label)

    # Plot the best fit regression line
    plt.plot(x_values, regression_y_values, linewidth=3, linestyle='--', color='r', label='Regression')

    # Add titles and labels
    plt.title(graph_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Rotate x-axis labels for readability
    plt.xticks(x_values, rotation=x_rotation)

    # Set y-axis ticks to divide into 20 parts
    y_min, y_max = min(y_values), max(y_values)
    y_ticks = np.linspace(y_min, y_max, 20)
    plt.yticks(y_ticks)

    plt.grid(grid)
    plt.legend()
    plt.show()

