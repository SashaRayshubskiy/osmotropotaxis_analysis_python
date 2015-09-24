__author__ = 'sasha'


# Load the data into the environment
import osmotropotaxis_main
d = osmotropotaxis_main.osmotropotaxis_main()
d.run()
# Init analyzer
import fly_trajectory_analyzer as fta
fly_traj_analyzer = fta.fly_trajectory_analyzer(d.exp_mdata, d.loader.trial_data_all)



# Show analysis, new analysis, new method
fly_traj_analyzer.show_avg_velocity_response()

# Show classifier
fly_traj_analyzer.show_classifier('AgglomerativeClustering')

#
fly_traj_analyzer.show_fwd_and_lat_velocity_bar_plots_for_each_trial()

#
fly_traj_analyzer.show_trajectories()