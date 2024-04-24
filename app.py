import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Constants
NUM_SIMULATIONS = 1000
CONFIDENCE_LEVEL = 0.95

# Set the page config to wide mode
st.set_page_config(layout="wide")

# Initialize the random generator
rng = np.random.default_rng(42)

# Function that works similarly to the Excel model
def excel_like_simulation(detection_size_avg, wind_speed_avg, response_time_avg, num_simulations):
    fire_sizes_acres = []
    for _ in range(num_simulations):
        detection_size = abs(np.random.normal(detection_size_avg, 0.5))
        wind_speed = np.random.normal(wind_speed_avg, 0.622)
        response_time = np.random.normal(response_time_avg, 10)
        
        # Convert detection size from acres to square meters
        detection_size_m2 = detection_size * 4046.86
        
        # Calculate the radius of the fire detection area
        detection_radius_m = np.sqrt(detection_size_m2 / np.pi)
        
        # Estimate fire spread and suppression effectiveness
        fire_spread_rate = wind_speed * 0.05
        fire_suppression_rate = detection_radius_m * 1.005
        
        # Calculate the final size of the fire
        burn_time_hrs = response_time / 3600
        final_radius_m = detection_radius_m + (fire_spread_rate * burn_time_hrs) - fire_suppression_rate
        final_size_acres = (final_radius_m**2 * np.pi) / 4046.86
        
        fire_sizes_acres.append(final_size_acres)
        
    return np.mean(fire_sizes_acres), np.std(fire_sizes_acres), fire_sizes_acres

# Function to simulate fire
def simulate_fire(num_simulations, detection_size_input, wind_speed_input, response_time_input):
    results = []
    fire_sizes = []

    for _ in range(num_simulations):
        # Use the user inputs for mean values
        detection_size_mean = detection_size_input
        wind_speed_mean = wind_speed_input
        response_time_mean = response_time_input

        # Standard deviations based on a hypothetical previous definition
        detection_size_std = 0.5
        wind_speed_std = 0.622
        response_time_std = 10

        # Other parameters are calculated based on inputs
        fire_spread = 0.1 * wind_speed_mean
        start_fighting_radius = np.sqrt(detection_size_mean * 4046.86 / np.pi)
        fire_suppression = start_fighting_radius * 1.001

        # Simulation using normal distribution
        detection_size = abs(rng.normal(detection_size_mean, detection_size_std))
        wind_speed = rng.normal(wind_speed_mean, wind_speed_std)
        response_time = rng.normal(response_time_mean, response_time_std)
        burn_time = response_time  # Already in seconds

        # Calculate the final radius and size after fire spread and suppression
        final_radius = start_fighting_radius + fire_spread * burn_time - fire_suppression
        if final_radius < 0:
            final_radius = 0
        final_size = np.pi * (final_radius ** 2)
        fire_sizes.append(final_size)

        results.append({
            'Detection Size (acres)': detection_size,
            'Wind Speed (m/s)': wind_speed,
            'Response Time (sec)': response_time,
            'Burn Time (sec)': burn_time,
            'Final Radius (m)': final_radius,
            'Final Size (m^2)': final_size
        })

    return pd.DataFrame(results), fire_sizes

# Main function
def main():
    # Title of the application
    st.title("Fire Simulation Dashboard")

    # Sidebar for input parameters
    st.sidebar.header("Input Parameters")
    st.sidebar.markdown("Adjust the following parameters to customize the simulation:")
    detection_size_input = st.sidebar.slider('Detection Size (acres)', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    wind_speed_input = st.sidebar.slider('Wind Speed (m/s)', min_value=0.1, max_value=20.0, value=6.19, step=0.01)
    response_time_input = st.sidebar.slider('Response Time (sec)', min_value=1, max_value=600, value=320, step=1)
    num_simulations = st.sidebar.number_input('Number of Simulations', min_value=100, max_value=10000, value=NUM_SIMULATIONS)

    # Button to run simulation
    if st.sidebar.button("Run Simulation"):
        results_df, fire_sizes = simulate_fire(num_simulations, detection_size_input, wind_speed_input, response_time_input)
        
        # Display simulation results
        st.subheader("Simulation Results")
        st.markdown("The table below shows the results of the fire simulation based on the input parameters.")
        st.dataframe(results_df.style.format("{:.2f}"))

        # Confidence Intervals for Final Size
        mean_size = np.mean(fire_sizes)
        std_err = stats.sem(fire_sizes)
        ci = stats.t.interval(CONFIDENCE_LEVEL, len(fire_sizes)-1, loc=mean_size, scale=std_err)
        st.subheader("Confidence Intervals for Final Fire Size:")
        st.markdown(f"The **confidence interval** represents the range in which the true average final fire size is likely to fall. With a confidence level of {CONFIDENCE_LEVEL}, it is estimated that the true mean final fire size lies within this interval.")
        st.write(f"Lower Bound: {ci[0]:.2f} m^2")
        st.write(f"Upper Bound: {ci[1]:.2f} m^2")

        # Correlation Matrix
        correlation_matrix = results_df.corr()
        st.subheader("Correlation Matrix:")
        st.markdown("The correlation matrix displays the relationships between different input parameters and the final fire size. A value close to 1 or -1 indicates a strong correlation, while a value close to 0 indicates no correlation.")
        st.write(correlation_matrix)

        # Sensitivity Analysis
        sensitivity = results_df.drop(columns=['Final Size (m^2)']).apply(lambda x: x.corr(results_df['Final Size (m^2)']))
        st.subheader("Sensitivity of Final Size to Input Parameters:")
        st.markdown("The sensitivity analysis measures how changes in each input parameter affect the final fire size. A higher sensitivity value indicates that changes in that parameter have a larger impact on the final size.")
        st.write(sensitivity.to_frame(name='Sensitivity'))

        # Generate and display plots
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        results_df.plot(x='Burn Time (sec)', y='Final Size (m^2)', kind='scatter', ax=axs[0], color='red', title='Fire Size Over Time')
        results_df['Final Radius (m)'].hist(bins=30, ax=axs[1], color='blue')
        axs[1].set_title('Final Radius Distribution')
        axs[1].set_xlabel('Final Radius (m)')
        axs[1].set_ylabel('Frequency')
        for ax in axs:
            ax.grid(True)
        st.subheader("Plots")
        st.markdown("The plots below visualize the progression of fire size over time and the distribution of final radii.")
        st.pyplot(fig)

        # # Real-time Fire Visualization
        # st.subheader("Real-time Fire Visualization")
        # st.markdown("The animation below demonstrates the progression of the fire in real-time, represented by a 3D scatter plot.")
        # st.plotly_chart(visualize_fire2(results_df))

if __name__ == "__main__":
    main()
