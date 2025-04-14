import os
from gtda.time_series import SingleTakensEmbedding
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy, Amplitude, NumberOfPoints, BettiCurve

from gtda.plotting import plot_diagram
from gtda.plotting import plot_point_cloud

from gtda.mapper import (
    CubicalCover,
    make_mapper_pipeline,
    Projection,
    plot_static_mapper_graph,
    plot_interactive_mapper_graph
)

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

def create_point_cloud(time_series, dimension, time_delay, stride):

    embedder = SingleTakensEmbedding(
        parameters_type="fixed",
        n_jobs=2,
        time_delay=time_delay,
        dimension=dimension,
        stride=stride,
    )

    return embedder.fit_transform(time_series)

def generate_plot_point_cloud(point_cloud, initial_slice_position, slice_size, save, path_to_save, filename):

    # Define axis configurations as dictionaries
    xaxis_config = {
        'range': [0, 1],
        # 'type': 'linear',  # or 'log'
        # 'autorange': False,
        # 'exponentformat': 'e',
        # 'tickformat': ".2f"
    }

    yaxis_config = {
        'range': [0, 1],
        # 'type': 'linear',  # or 'log'
        # 'autorange': False,
        # 'scaleanchor': 'x',
        # 'scaleratio': 1,
        # 'exponentformat': 'E',
        # 'tickformat': ".2f"
    }

    # define the title
    title=f'Point Cloud | Initial position: {initial_slice_position} | Slice size: {slice_size}'

    layout_config_fix_axis = {
        'xaxis': xaxis_config,
        'yaxis': yaxis_config,
        'title': title,
        'width': 500,
        'height':500
    }

    layout_config = {
        'title': title,
        'width': 500,
        'height':500
    }    

    fig_fix_axis = plot_point_cloud(point_cloud, plotly_params=dict(layout=layout_config_fix_axis))
    fig = plot_point_cloud(point_cloud, plotly_params=dict(layout=layout_config))

    if save:
        fig_fix_axis.write_image(os.path.join(path_to_save, filename + '_fixed_axis.png'))
        fig.write_image(os.path.join(path_to_save, filename + '.png'))

    return fig


def generate_plot_persistence_diagram(point_cloud, initial_slice_position, slice_size, save, path_to_save, filename):

    # Create an instance of VietorisRipsPersistence
    VR = VietorisRipsPersistence(metric='euclidean', homology_dimensions=[0, 1, 2])

    # Fit and transform your data
    diagrams = VR.fit_transform(point_cloud.reshape(1, *point_cloud.shape))    

    # Define axis configurations as dictionaries
    xaxis_config = {
        'range': [0, 0.05],
        # 'type': 'linear',  # or 'log'
        # 'autorange': False,
        # 'exponentformat': 'e',
        # 'tickformat': ".2f"
    }

    yaxis_config = {
        'range': [0, 0.05],
        # 'type': 'linear',  # or 'log'
        # 'autorange': False,
        # 'scaleanchor': 'x',
        # 'scaleratio': 1,
        # 'exponentformat': 'E',
        # 'tickformat': ".2f"
    }

    # define the title
    title = f'Persistent Diagram | Initial position: {initial_slice_position} | Slice size: {slice_size}'

    layout_config_fixed_axis = {
        'xaxis': xaxis_config,
        'yaxis': yaxis_config,
        'title': title,
        'width': 500,
        'height':500
    }

    layout_config = {
        'title': title,
        'width': 500,
        'height':500
    }    

    fig_fixex_axis = plot_diagram(diagrams[0], plotly_params=dict(layout=layout_config_fixed_axis))
    fig = plot_diagram(diagrams[0], plotly_params=dict(layout=layout_config))

    if save:
        fig_fixex_axis.write_image(os.path.join(path_to_save, filename + '_fixed_axis.png'))
        fig.write_image(os.path.join(path_to_save, filename + '.png'))

    return fig


def generate_plot_mapper_graph(point_cloud, initial_slice_position, slice_size, save, path_to_save, filename):

    # Define filter function – can be any scikit-learn transformer
    filter_func = Projection(columns=[0, 1])
    # Define cover
    cover = CubicalCover(n_intervals=10, overlap_frac=0.3)
    # Choose clustering algorithm – default is DBSCAN
    clusterer = DBSCAN()

    # Configure parallelism of clustering step
    n_jobs = 1

    # Initialise pipeline
    pipe = make_mapper_pipeline(
        filter_func=filter_func,
        cover=cover,
        clusterer=clusterer,
        verbose=False,
        n_jobs=n_jobs,
    )

    # define the title
    title = f'Mapper Graph | Initial position: {initial_slice_position} | Slice size: {slice_size}'

    layout_config = {
        'title': title,
        'width': 500,
        'height':500
    }

    fig = plot_static_mapper_graph(pipe, point_cloud, plotly_params=dict(layout=layout_config))
    
    if save:
        fig.write_image(os.path.join(path_to_save, filename + '.png'))
    
    return fig


def get_features(point_cloud):

    # Create an instance of VietorisRipsPersistence
    # 0 - connected components, 1 - loops, 2 - voids
    VR = VietorisRipsPersistence(metric='euclidean', homology_dimensions=[0, 1, 2])

    # Fit and transform your data
    diagrams = VR.fit_transform(point_cloud.reshape(1, *point_cloud.shape))

    # Create instances of the feature extraction objects
    PE = PersistenceEntropy()
    Amp = Amplitude(metric='landscape')
    NP = NumberOfPoints()
    BC = BettiCurve()

    # Extract features from the diagrams
    entropy_features = PE.fit_transform(diagrams)
    amplitude_features = Amp.fit_transform(diagrams)
    number_of_points_features = NP.fit_transform(diagrams)
    betti_curve_features = BC.fit_transform(diagrams)

    # get entropy features into a dictionary
    # entropy_features[0] - connected components, entropy_features[1] - loops, entropy_features[2] - voids 

    entropy_features_dict = {
        'connected_components_entropy': entropy_features[0][0],
        'loops_entropy': entropy_features[0][1],
        'voids_entropy': entropy_features[0][2]
    }

    # get amplitude features into a dictionary
    # amplitude_features[0] - connected components, amplitude_features[1] - loops, amplitude_features[2] - voids

    amplitude_features_dict = {
        'connected_components_amplitude': amplitude_features[0][0],
        'loops_amplitude': amplitude_features[0][1],
        'voids_amplitude': amplitude_features[0][2]
    }

    # get number of points features into a dictionary
    # number_of_points_features[0] - connected components, number_of_points_features[1] - loops, number_of_points_features[2] - voids

    number_of_points_features_dict = {
        'connected_components_number_of_points': number_of_points_features[0][0],
        'loops_number_of_points': number_of_points_features[0][1],
        'voids_number_of_points': number_of_points_features[0][2]
    }

    return entropy_features_dict, amplitude_features_dict, number_of_points_features_dict

