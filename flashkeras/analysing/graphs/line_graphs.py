from flashkeras.utils.otherimports import *
from flashkeras.utils.typehints import *
from sklearn.metrics import mean_squared_error # type: ignore

def plot_line_graph(y_values: list | np.ndarray, 
                    x_values: list | Any = None,
                    graph_title='Line Graph',
                    x_label='X', 
                    y_label='Y',
                    x_rotation=90,
                    fig_size=(15, 6),
                    x_ticks=None,
                    y_ticks=None,
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

    if x_ticks is None:
        x_ticks = x_values
    plt.xticks(x_ticks, rotation=x_rotation)

    if y_ticks is None:
        y_min, y_max = min(y_values), max(y_values)
        y_ticks = np.linspace(y_min, y_max, 20)
    plt.yticks(y_ticks)

    # Disable grid, add legend, and show plot
    plt.grid(grid)
    plt.legend()
    plt.show()

def get_line_graph(y_values: list | np.ndarray, 
                   x_values: list | Any = None,
                   graph_title='Line Graph',
                   x_label='X', 
                   y_label='Y',
                   x_rotation=90,
                   fig_size=(15, 6),
                   x_ticks=None,
                   y_ticks=None,
                   grid: bool = False):
    
    if x_values is None:
        x_values = np.arange(1, len(y_values) + 1)

    fig, ax = plt.subplots(figsize=fig_size)
    
    ax.plot(x_values, y_values, marker='o', linestyle='-', color='b', label=x_label)

    ax.set_title(graph_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if x_ticks is None:
        x_ticks = x_values
    ax.set_xticks(x_ticks)
    ax.tick_params(axis='x', rotation=x_rotation)

    if y_ticks is None:
        y_min, y_max = min(y_values), max(y_values)
        y_ticks = np.linspace(y_min, y_max, 20)
    ax.set_yticks(y_ticks)

    ax.grid(grid)
    ax.legend()

    return fig

def plot_multi_line_graph(coords: list[list[list]], 
                          x_values: list = None,
                          graph_title='Line Graph',
                          x_label='X', 
                          y_label='Y',
                          line_labels: list | None | Literal['default'] = None, 
                          x_rotation=90,
                          fig_size=(15, 6),
                          x_ticks=None,
                          y_ticks=None,
                          grid: bool = False):
    """Plots multiple lines on the same graph with the given coordnates
    
    :param coords: List of coordnates lists i.e. [ [[x, y], [x, y]], [[x, y], [x, y], [x, y]] ].
    :param labels: Names of the different lines.
    """
    
    if x_values is None:
        x_values = [list(zip(*coord))[0] for coord in coords]
    
    y_values = [list(zip(*coord))[1] for coord in coords]

    plt.figure(figsize=fig_size)

    if line_labels == 'default':
        line_labels = [f'Line {i+1}' for i in range(len(coords))]

    for i, (x, y) in enumerate(zip(x_values, y_values)):
        if line_labels is not None:
            plt.plot(x, y, marker='o', linestyle='-', label=line_labels[i])
        else:
            plt.plot(x, y, marker='o', linestyle='-')

    plt.title(graph_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if x_ticks is None:
        x_ticks = np.concatenate(x_values)
    plt.xticks(x_ticks, rotation=x_rotation)

    if y_ticks is None:
        all_y_values = np.concatenate(y_values)
        y_min, y_max = min(all_y_values), max(all_y_values)
        y_ticks = np.linspace(y_min, y_max, 20)
    plt.yticks(y_ticks)

    plt.grid(grid)
    plt.legend()

    plt.show()

def get_multi_line_graph(coords: list[list[list]], 
                         x_values: list = None,
                         graph_title='Line Graph',
                         x_label='X', 
                         y_label='Y',
                         line_labels: list | None | Literal['default'] = None, 
                         x_rotation=90,
                         fig_size=(15, 6),
                         x_ticks=None,
                         y_ticks=None,
                         grid: bool = False):
    """Generate a multi-lines graph with the given coordnates and returns it
    
    :param coords: List of coordnates lists i.e. [ [[x, y], [x, y]], [[x, y], [x, y], [x, y]] ].
    :param labels: Names of the different lines.
    """
    
    if x_values is None:
        x_values = [list(zip(*coord))[0] for coord in coords]
    
    y_values = [list(zip(*coord))[1] for coord in coords]

    fig, ax = plt.subplots(figsize=fig_size)

    if line_labels == 'default':
        line_labels = [f'Line {i+1}' for i in range(len(coords))]

    for i, (x, y) in enumerate(zip(x_values, y_values)):
        if line_labels is not None:
            plt.plot(x, y, marker='o', linestyle='-', label=line_labels[i])
        else:
            plt.plot(x, y, marker='o', linestyle='-')

    ax.set_title(graph_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if x_ticks is None:
        x_ticks = np.concatenate(x_values)
    ax.set_xticks(x_ticks)
    ax.tick_params(axis='x', rotation=x_rotation)

    if y_ticks is None:
        all_y_values = np.concatenate(y_values)
        y_min, y_max = min(all_y_values), max(all_y_values)
        y_ticks = np.linspace(y_min, y_max, 20)
    ax.set_yticks(y_ticks)

    ax.grid(grid)
    ax.legend()

    return fig

def plot_multi_line_graph_polynomial_fit(coords: list[list[list]], 
                          max_degree=5,
                          graph_title='Line Graph',
                          x_label='X', 
                          y_label='Y',
                          line_labels: list | None | Literal['default'] = None, 
                          x_rotation=90,
                          fig_size=(15, 6),
                          x_ticks=None,
                          y_ticks=None,
                          grid: bool = False,
                          regression_line_color='black'
                          ):
    """Plots multiple lines on the same graph with the given coordnates and also plots the polynomial regression of all the points.
    
    :param coords: List of coordnates lists i.e. [ [[x, y], [x, y]], [[x, y], [x, y], [x, y]] ].
    :param labels: Names of the different lines.
    """
    
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
    
    x_values_flat = np.concatenate([list(zip(*coord))[0] for coord in coords])
    y_values_flat = np.concatenate([list(zip(*coord))[1] for coord in coords])

    plt.figure(figsize=fig_size)

    best_degree = find_best_polynomial_degree(x_values_flat, y_values_flat, max_degree)

    coefficients = np.polyfit(x_values_flat, y_values_flat, best_degree)
    polynomial = np.poly1d(coefficients)
    regression_y_values = polynomial(np.sort(x_values_flat))

    if line_labels == 'default':
        line_labels = [f'Line {i+1}' for i in range(len(coords))]

    for i, coord in enumerate(coords):
        x, y = zip(*coord)
        if line_labels is not None:
            plt.plot(x, y, marker='o', linestyle='-', label=line_labels[i])
        else:
            plt.plot(x, y, marker='o', linestyle='-')

    plt.plot(np.sort(x_values_flat), regression_y_values, linewidth=3, linestyle='--', color=regression_line_color, label=f'Regression (Degree {best_degree})')

    plt.title(graph_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if x_ticks is None:
        x_ticks = np.sort(np.unique(x_values_flat))
    plt.xticks(x_ticks, rotation=x_rotation)

    if y_ticks is None:
        y_min, y_max = min(y_values_flat), max(y_values_flat)
        y_ticks = np.linspace(y_min, y_max, 20)
    plt.yticks(y_ticks)

    plt.grid(grid)
    plt.legend()

    plt.show()

def get_multi_line_graph_polynomial_fit(coords: list[list[list]], 
                          max_degree=5,
                          graph_title='Line Graph',
                          x_label='X', 
                          y_label='Y',
                          line_labels: list | None | Literal['default'] = None, 
                          x_rotation=90,
                          fig_size=(15, 6),
                          x_ticks=None,
                          y_ticks=None,
                          grid: bool = False):
    """Generate a multi-lines graph with the given coordnates and a polynomial regression and returns it
    
    :param coords: List of coordnates lists i.e. [ [[x, y], [x, y]], [[x, y], [x, y], [x, y]] ].
    :param labels: Names of the different lines.
    """
    
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
    
    x_values_flat = np.concatenate([list(zip(*coord))[0] for coord in coords])
    y_values_flat = np.concatenate([list(zip(*coord))[1] for coord in coords])

    fig, ax = plt.subplots(figsize=fig_size)

    best_degree = find_best_polynomial_degree(x_values_flat, y_values_flat, max_degree)

    coefficients = np.polyfit(x_values_flat, y_values_flat, best_degree)
    polynomial = np.poly1d(coefficients)
    regression_y_values = polynomial(np.sort(x_values_flat))

    if line_labels == 'default':
        line_labels = [f'Line {i+1}' for i in range(len(coords))]

    for i, coord in enumerate(coords):
        x, y = zip(*coord)
        if line_labels is not None:
            plt.plot(x, y, marker='o', linestyle='-', label=line_labels[i])
        else:
            plt.plot(x, y, marker='o', linestyle='-')

    ax.plot(np.sort(x_values_flat), regression_y_values, linewidth=3, linestyle='--', color='b', label=f'Regression (Degree {best_degree})')

    ax.set_title(graph_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if x_ticks is None:
        x_ticks = np.sort(np.unique(x_values_flat))
    ax.set_xticks(x_ticks)
    ax.tick_params(axis='x', rotation=x_rotation)

    if y_ticks is None:
        y_min, y_max = min(y_values_flat), max(y_values_flat)
        y_ticks = np.linspace(y_min, y_max, 20)
    ax.set_yticks(y_ticks)

    ax.grid(grid)
    ax.legend()

    return fig

def plot_line_graph_polynomial_fit_zoom_higher_y(y_values: list | np.ndarray, 
                                               x_values: list | Any = None,
                                               max_degree=5, 
                                               focus_threshold=0.8,
                                               graph_title='Graph with Polynomial Regression & Y axis Cut',
                                               x_label='X', 
                                               y_label='Y',
                                               x_rotation=90,
                                               fig_size=(15, 6),
                                               x_ticks=None,
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

    best_degree = find_best_polynomial_degree(x_values, y_values, max_degree)

    coefficients = np.polyfit(x_values, y_values, best_degree)
    polynomial = np.poly1d(coefficients)
    regression_y_values = polynomial(x_values)

    plt.figure(figsize=fig_size)
    
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', label=x_label)

    plt.plot(x_values, regression_y_values, linewidth=3, linestyle='--', color='r', label='Regression')

    plt.title(graph_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if x_ticks is None:
        x_ticks = x_values
        
    plt.xticks(x_ticks, rotation=x_rotation)

    min_val, max_val = find_y_limits(y_values)
    plt.ylim(min_val, max_val)

    plt.grid(grid)
    plt.legend()
    plt.show()

def get_line_graph_polynomial_fit_zoom_higher_y(y_values: list | np.ndarray, 
                                                x_values: list | Any = None,
                                                max_degree=5, 
                                                focus_threshold=0.8,
                                                graph_title='Graph with Polynomial Regression & Y axis Cut',
                                                x_label='X', 
                                                y_label='Y',
                                                x_rotation=90,
                                                fig_size=(15, 6),
                                                x_ticks=None,
                                                grid: bool = False) -> plt.Figure:
    
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

    best_degree = find_best_polynomial_degree(x_values, y_values, max_degree)

    coefficients = np.polyfit(x_values, y_values, best_degree)
    polynomial = np.poly1d(coefficients)
    regression_y_values = polynomial(x_values)

    fig, ax = plt.subplots(figsize=fig_size)
    
    ax.plot(x_values, y_values, marker='o', linestyle='-', color='b', label=x_label)

    ax.plot(x_values, regression_y_values, linewidth=3, linestyle='--', color='r', label='Regression')

    ax.set_title(graph_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if x_ticks is None:
        x_ticks = x_values
    ax.set_xticks(x_ticks)
    ax.tick_params(axis='x', rotation=x_rotation)

    min_val, max_val = find_y_limits(y_values)
    ax.set_ylim(min_val, max_val)

    ax.grid(grid)
    ax.legend()

    return fig

def plot_line_graph_polynomial_fit(y_values: list | np.ndarray, 
                                   x_values: list | Any = None,
                                   max_degree=5, 
                                   graph_title='Graph with Polynomial Regression',
                                   x_label='X', 
                                   y_label='Y',
                                   x_rotation=90,
                                   fig_size=(15, 6),
                                   x_ticks=None,
                                   y_ticks=None,
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

    if x_ticks is None:
        x_ticks = x_values

    plt.xticks(x_ticks, rotation=x_rotation)

    if y_ticks is None:
        y_min, y_max = min(y_values), max(y_values)
        y_ticks = np.linspace(y_min, y_max, 20)
    plt.yticks(y_ticks)

    plt.grid(grid)
    plt.legend()
    plt.show()

def get_line_graph_polynomial_fit(y_values: list | np.ndarray, 
                                  x_values: list = None,
                                  max_degree=5, 
                                  graph_title='Graph with Polynomial Regression',
                                  x_label='X', 
                                  y_label='Y',
                                  x_rotation=90,
                                  fig_size=(7, 4),  # Tamanho da figura ajustado para ser mais apropriado para Tkinter
                                  x_ticks=None,
                                  y_ticks=None,
                                  grid: bool = False):

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

    best_degree = find_best_polynomial_degree(x_values, y_values, max_degree)

    coefficients = np.polyfit(x_values, y_values, best_degree)
    polynomial = np.poly1d(coefficients)
    regression_y_values = polynomial(x_values)

    fig, ax = plt.subplots(figsize=fig_size)
    
    ax.plot(x_values, y_values, marker='o', linestyle='-', color='b', label='Dados Originais')

    ax.plot(x_values, regression_y_values, linewidth=3, linestyle='--', color='r', label='Regressão Polinomial')

    # Define título e rótulos
    ax.set_title(graph_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if x_ticks is None:
        x_ticks = x_values
    ax.set_xticks(x_ticks)
    ax.tick_params(axis='x', rotation=x_rotation)

    if y_ticks is None:
        y_min, y_max = min(y_values), max(y_values)
        y_ticks = np.linspace(y_min, y_max, 20)
    ax.set_yticks(y_ticks)

    ax.grid(grid)
    ax.legend()

    return fig
