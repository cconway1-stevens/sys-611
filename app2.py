import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Constants
DEFAULT_SIMULATIONS = 3
FOREST_SIZE = 50  # Grid size for the 2D forest simulation

# Set the page config to wide mode
st.set_page_config(layout="wide")


# Function to simulate fire
def simulate_fire(detection_size, wind_speed, response_time):
    forest = np.zeros((FOREST_SIZE, FOREST_SIZE))
    fire_center = FOREST_SIZE // 2
    forest[fire_center][fire_center] = detection_size * 100  # Start fire in the middle

    for _ in range(response_time):
        spread_rate = wind_speed / 10
        new_forest = forest.copy()
        for row in range(FOREST_SIZE):
            for col in range(FOREST_SIZE):
                if forest[row][col] > 0:
                    new_intensity = forest[row][col] * (1 - spread_rate)  # Decay fire intensity
                    # Spread fire to neighbors
                    if row > 0:
                        new_forest[row - 1][col] += new_intensity / 4
                    if row < FOREST_SIZE - 1:
                        new_forest[row + 1][col] += new_intensity / 4
                    if col > 0:
                        new_forest[row][col - 1] += new_intensity / 4
                    if col < FOREST_SIZE - 1:
                        new_forest[row][col + 1] += new_intensity / 4
        forest = new_forest
    return forest


# Function to visualize the fire in a heatmap
def visualize_fire(forest):
    fig = go.Figure(data=go.Heatmap(
        z=forest,
        colorscale='Reds',
        showscale=False
    ))
    fig.update_layout(title="Fire Spread Simulation", width=800, height=800)
    return fig


# Function to run simulations and visualize results
def run_simulation(detection_size, wind_speed, response_time, num_simulations):
    simulations_run = 0
    while simulations_run < num_simulations:
        forest = simulate_fire(detection_size, wind_speed, response_time)
        st.subheader(f"Simulation {simulations_run + 1} Results")
        st.plotly_chart(visualize_fire(forest), use_container_width=True)
        # Display final forest state after each simulation
        st.text("Final Forest State:")
        st.write(forest)
        simulations_run += 1


# Main function
def main():
    st.title("2D Fire Simulation Game")

    # Sidebar for input parameters
    st.sidebar.markdown("**Fire Parameters**")
    detection_size = st.sidebar.slider('Detection Size (affects initial intensity)', 0.1, 5.0, 1.0, 0.1)
    wind_speed = st.sidebar.slider('Wind Speed (m/s)', 1, 15, 5, 1)
    response_time = st.sidebar.slider('Response Time (seconds)', 10, 100, 50, 10)
    num_simulations = st.sidebar.slider('Number of Simulations', 1, 10, DEFAULT_SIMULATIONS, 1)

    # Buttons for simulation control
    run_button = st.sidebar.button("Run Simulation")
    stop_button = st.sidebar.button("Stop Simulation", disabled=True)

    if run_button:
        run_simulation(detection_size, wind_speed, response_time, num_simulations)
    elif stop_button:
        st.sidebar.warning("Stopping simulation is not currently implemented.")


if __name__ == "__main__":
    main()
