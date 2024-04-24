import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fire_simulation import simulate_fire, calculate_confidence_interval, calculate_correlation_matrix, calculate_sensitivity

# Constants
NUM_SIMULATIONS = 1000
CONFIDENCE_LEVEL = 0.95

# Set the page config to wide mode
st.set_page_config(layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .st-eb {
        background-color: #f0f0f0;
    }
    .st-dl {
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0px 0px 5px 0px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .st-cq {
        border-radius: 10px;
        box-shadow: 0px 0px 5px 0px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    st.title("Fire Simulation Dashboard")

    # Sidebar for user input
    st.sidebar.header("Input Parameters")
    detection_size_input = st.sidebar.slider('Detection Size (acres)', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    wind_speed_input = st.sidebar.slider('Wind Speed (m/s)', min_value=0.1, max_value=20.0, value=6.19, step=0.01)
    response_time_input = st.sidebar.slider('Response Time (sec)', min_value=1, max_value=600, value=320, step=1)
    num_simulations = st.sidebar.number_input(
        'Number of Simulations',
        min_value=100,
        max_value=10000,
        value=NUM_SIMULATIONS
    )

    if st.sidebar.button("Run Simulation"):
        # Run simulations
        results_df, fire_sizes = simulate_fire(num_simulations, detection_size_input, wind_speed_input, response_time_input)
        
        with st.container():
            # Simulation Results
            st.subheader("Simulation Results")
            st.write("This section displays the results of the fire simulations. It provides details such as detection size, wind speed, response time, burn time, final radius, and final size for each simulation.")

            # Display the dataframe
            st.dataframe(results_df.style.format("{:.2f}"))

        # Confidence Intervals for Final Size
        lower_bound, upper_bound = calculate_confidence_interval(fire_sizes, CONFIDENCE_LEVEL)

        st.subheader("Confidence Intervals for Final Fire Size:")
        st.write(f"Lower Bound: {lower_bound:.2f} m^2")
        st.write(f"Upper Bound: {upper_bound:.2f} m^2")
        st.write("These confidence intervals represent the range in which the true mean final fire size is likely to fall with 95% confidence. They help assess the uncertainty in our simulations.")

        # Correlation Analysis
        correlation_matrix = calculate_correlation_matrix(results_df)
        st.subheader("Correlation Matrix:")
        st.write("The correlation matrix shows the correlation coefficients between different variables. It helps understand how changes in one variable may affect another.")

        # Display the correlation matrix
        st.write(correlation_matrix)

        # Sensitivity Analysis
        sensitivity = calculate_sensitivity(results_df)
        st.subheader("Sensitivity of Final Size to Input Parameters:")
        st.write("This section quantifies the sensitivity of the final fire size to the input parameters. It helps identify which parameters have the most significant impact on the outcome.")

        # Display sensitivity analysis
        st.write(sensitivity)

        # Generate plots
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        results_df.plot(x='Burn Time (sec)', y='Final Size (m^2)', kind='scatter', ax=axs[0], color='red', title='Fire Size Over Time')
        results_df['Final Radius (m)'].hist(bins=30, ax=axs[1], color='blue')
        axs[1].set_title('Final Radius Distribution')
        axs[1].set_xlabel('Final Radius (m)')
        axs[1].set_ylabel('Frequency')

        for ax in axs:
            ax.grid(True)

        st.subheader("Plots")
        st.write("These plots visualize the fire size over time and the distribution of final radii across simulations. They provide additional insights into the behavior and characteristics of the fire.")

        # Display plots
        st.pyplot(fig)

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import plotly.express as px
# Constants
NUM_SIMULATIONS = 1000
CONFIDENCE_LEVEL = 0.95

# Set the page config to wide mode
st.set_page_config(layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .st-eb {
        background-color: #f0f0f0;
    }
    .st-dl {
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0px 0px 5px 0px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .st-cq {
        border-radius: 10px;
        box-shadow: 0px 0px 5px 0px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize the random generator
rng = np.random.default_rng()

def simulate_fire(num_simulations, detection_size_input, wind_speed_input, response_time_input):
    results = []
    fire_sizes = []

    for _ in range(num_simulations):
        # Use the user inputs for mean values
        detection_size_mean = detection_size_input
        wind_speed_mean = wind_speed_input
        response_time_mean = response_time_input
        
        # Standard deviations are fixed based on the spreadsheet
        detection_size_std = 0.5
        wind_speed_std = 0.622
        response_time_std = 10
        
        # Other parameters are calculated in the spreadsheet
        fire_spread = 0.1 * wind_speed_mean
        start_fighting_radius = np.sqrt(detection_size_mean * 4046.86 / np.pi)
        fire_suppression = start_fighting_radius * 1.001

        # Simulation using normal distribution
        detection_size = abs(rng.normal(detection_size_mean, detection_size_std))
        wind_speed = rng.normal(wind_speed_mean, wind_speed_std)
        response_time = rng.normal(response_time_mean, response_time_std)
        
        # Convert response time to burn time
        burn_time = response_time  # Already in seconds according to the spreadsheet
        
        # Calculate initial fire fighting size
        initial_fighting_size = np.pi * (start_fighting_radius**2)

        # Calculate the final radius after fire spread and suppression
        final_radius = start_fighting_radius + fire_spread * burn_time - fire_suppression
        if final_radius < 0:
            final_radius = 0
        
        # Calculate final size
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

def main():
    st.title("Fire Simulation Dashboard")

    # Sidebar for user input
    st.sidebar.header("Input Parameters")
    detection_size_input = st.sidebar.slider('Detection Size (acres)', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    wind_speed_input = st.sidebar.slider('Wind Speed (m/s)', min_value=0.1, max_value=20.0, value=6.19, step=0.01)
    response_time_input = st.sidebar.slider('Response Time (sec)', min_value=1, max_value=600, value=320, step=1)
    num_simulations = st.sidebar.number_input(
        'Number of Simulations',
        min_value=100,
        max_value=10000,
        value=NUM_SIMULATIONS
    )

    if st.sidebar.button("Run Simulation"):
        # Run simulations
        results_df, fire_sizes = simulate_fire(num_simulations, detection_size_input, wind_speed_input, response_time_input)
        
        with st.container():
            # Simulation Results
            st.subheader("Simulation Results")
            st.write("This section displays the results of the fire simulations. It provides details such as detection size, wind speed, response time, burn time, final radius, and final size for each simulation.")

            # Display the dataframe
            st.dataframe(results_df.style.format("{:.2f}"))

        # Confidence Intervals for Final Size
        mean_size = np.mean(fire_sizes)
        std_err = stats.sem(fire_sizes)
        ci = stats.t.interval(CONFIDENCE_LEVEL, len(fire_sizes), loc=mean_size, scale=std_err)

        st.subheader("Confidence Intervals for Final Fire Size:")
        st.write(f"Lower Bound: {ci[0]:.2f} m^2")
        st.write(f"Upper Bound: {ci[1]:.2f} m^2")
        st.write("These confidence intervals represent the range in which the true mean final fire size is likely to fall with 95% confidence. They help assess the uncertainty in our simulations.")

        # Correlation Analysis
        correlation_matrix = results_df.corr()
        st.subheader("Correlation Matrix:")
        st.write("The correlation matrix shows the correlation coefficients between different variables. It helps understand how changes in one variable may affect another.")

        # Display the correlation matrix
        st.write(correlation_matrix)

        # Sensitivity Analysis
        sensitivity = {}
        for column in results_df.columns[:-1]:  # Exclude the 'Final Size' column
            sensitivity[column] = np.corrcoef(results_df[column], results_df['Final Size (m^2)'])[0, 1]

        st.subheader("Sensitivity of Final Size to Input Parameters:")
        st.write("This section quantifies the sensitivity of the final fire size to the input parameters. It helps identify which parameters have the most significant impact on the outcome.")

        # Display sensitivity analysis
        st.write(sensitivity)

        # Generate plots
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        results_df.plot(x='Burn Time (sec)', y='Final Size (m^2)', kind='scatter', ax=axs[0], color='red', title='Fire Size Over Time')
        results_df['Final Radius (m)'].hist(bins=30, ax=axs[1], color='blue')
        axs[1].set_title('Final Radius Distribution')
        axs[1].set_xlabel('Final Radius (m)')
        axs[1].set_ylabel('Frequency')

        for ax in axs:
            ax.grid(True)

        st.subheader("Plots")
        st.write("These plots visualize the fire size over time and the distribution of final radii across simulations. They provide additional insights into the behavior and characteristics of the fire.")

        # Display plots
        st.pyplot(fig)


if __name__ == "__main__":
    main()
