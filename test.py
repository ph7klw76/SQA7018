from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import numpy as np

class SpatialPopulation:
    def __init__(self, size, density, p_infect, p_recover, p_vaccinate, p_susceptible_again, initial_infected=1):
        self.size = size
        self.density = density
        self.grid = np.full((size, size), -1)  # -1 for empty, 0: Susceptible, 1: Infected, 2: Recovered, 3: Vaccinated
        self.p_infect = p_infect
        self.p_recover = p_recover
        self.p_vaccinate = p_vaccinate
        self.p_susceptible_again = p_susceptible_again
        self.populate_grid(initial_infected)

    def populate_grid(self, initial_infected):
        n_individuals = int(self.size**2 * self.density)
        indices = np.random.choice(self.size**2, n_individuals, replace=False)
        self.grid.flat[indices] = 0  # Mark as susceptible
        infected_indices = np.random.choice(indices, initial_infected, replace=False)
        self.grid.flat[infected_indices] = 1  # Initial infected individuals

    def get_neighbors(self, row, col):
        neighbors = [(i, j) for i in range(row-1, row+2) for j in range(col-1, col+2)
                     if 0 <= i < self.size and 0 <= j < self.size and (i != row or j != col)]
        return neighbors

    def update(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == 1:  # Infected
                    if np.random.random() < self.p_recover:
                        self.grid[i, j] = 2  # Recover
                elif self.grid[i, j] == 0:  # Susceptible
                    if np.random.random() < self.p_vaccinate:
                        self.grid[i, j] = 3  # Vaccinate
                    else:
                        neighbors = self.get_neighbors(i, j)
                        infected_neighbors = any(self.grid[x, y] == 1 for x, y in neighbors)
                        if infected_neighbors and np.random.random() < self.p_infect:
                            self.grid[i, j] = 1  # Infect
                elif self.grid[i, j] == 2:  # Recovered
                    if np.random.random() < self.p_susceptible_again:
                        self.grid[i, j] = 0  # Become susceptible again

    def count_states(self):
        counts = np.bincount(self.grid[self.grid >= 0], minlength=4)
        return counts  # Susceptible, Infected, Recovered, Vaccinated

# Simulation parameters
size = 50  # Grid size
density = 0.8  # Fraction of grid occupied by individuals
p_infect = 0.05
p_recover = 0.1
p_vaccinate = 0.01
p_susceptible_again = 0.003
n_steps = 1000
initial_infected = 10
population = SpatialPopulation(size, density, p_infect, p_recover, p_vaccinate, p_susceptible_again, initial_infected)

app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1("Interactive Spatial Population Model"),
    dcc.Graph(id='population-plot'),
    html.Button("Start", id="start-button", n_clicks=0),
    html.Button("Stop", id="stop-button", n_clicks=0),
    dcc.Interval(id='update-interval', interval=1000, n_intervals=0, disabled=True),
    dcc.Store(id='simulation-data'),  # Store for simulation state counts over time

    html.Div([
        html.Label("Density"),
        dcc.Slider(
            id='density-slider',
            min=0.0,
            max=1.0,
            value=0.8,
            step=0.01,
            marks={i / 10: str(i / 10) for i in range(0, 11)}
        ),

        html.Label("Infection Probability"),
        dcc.Slider(
            id='p_infect-slider',
            min=0.0,
            max=1.0,
            value=0.05,
            step=0.01,
            marks={i / 10: str(i / 10) for i in range(0, 11)}
        ),

        html.Label("Recovery Probability"),
        dcc.Slider(
            id='p_recover-slider',
            min=0.0,
            max=1.0,
            value=0.1,
            step=0.01,
            marks={i / 10: str(i / 10) for i in range(0, 11)}
        ),

        html.Label("Vaccination Probability"),
        dcc.Slider(
            id='p_vaccinate-slider',
            min=0.0,
            max=1.0,
            value=0.01,
            step=0.01,
            marks={i / 10: str(i / 10) for i in range(0, 11)}
        ),

        html.Label("Susceptible Again Probability"),
        dcc.Slider(
            id='p_susceptible_again-slider',
            min=0.0,
            max=1.0,
            value=0.003,
            step=0.001,
            marks={i / 100: f"{i / 100:.2f}" for i in range(0, 101, 5)}
        ),
    ], style={'padding': 20})
])

@app.callback(
    Output('update-interval', 'disabled'),
    [Input('start-button', 'n_clicks'),
     Input('stop-button', 'n_clicks')],
    [State('update-interval', 'disabled')]
)
def toggle_simulation(start_n, stop_n, is_disabled):
    if start_n > stop_n:
        return False  # Enable interval updates
    return True  # Disable interval updates

@app.callback(
    Output('simulation-data', 'data'),
    Input('update-interval', 'n_intervals'),
    State('simulation-data', 'data')
)
def update_simulation(n_intervals, data):
    if n_intervals == 0 or data is None:
        # Reset or initialize data storage
        data = {'time': [], 'susceptible': [], 'infected': [], 'recovered': [], 'vaccinated': []}
    else:
        # Update the model state
        population.update()
        counts = population.count_states()
        # Append new counts to the data storage
        data['time'].append(n_intervals)
        data['susceptible'].append(counts[0])
        data['infected'].append(counts[1])
        data['recovered'].append(counts[2])
        data['vaccinated'].append(counts[3])
    return data

@app.callback(
    Output('population-plot', 'figure'),
    Input('simulation-data', 'data')
)
def update_plot(data):
    if data is None:
        raise PreventUpdate
    # Create line graph for state counts over time
    fig = go.Figure()
    for state in ['susceptible', 'infected', 'recovered', 'vaccinated']:
        fig.add_trace(go.Scatter(x=data['time'], y=data[state], mode='lines', name=state.capitalize()))
    fig.update_layout(title="Population State Counts Over Time", xaxis_title="Time", yaxis_title="Count")
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
