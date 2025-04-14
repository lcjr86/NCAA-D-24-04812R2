import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import MinMaxScaler

import os

def get_sliced_time_series(df, initial_slice_position, slice_size):
    '''
    Get a dataframe with a specific column as input, the initial position of the slice and the size of the slice
    and return the series
    '''
    return df.iloc[initial_slice_position:initial_slice_position+slice_size]


def generate_plot_sliced_time_series(df, initial_slice_position, slice_size, save, path_to_save, filename):
    '''
    Get a dataframe with a specific column as input, the initial position of the slice and the size of the slice
    and plot the series
    '''
    time_series = get_sliced_time_series(df, initial_slice_position, slice_size)

    # # Define axis configurations as dictionaries
    # xaxis_config = {
    #     'range': [0, 0.05],
    #     # 'type': 'linear',  # or 'log'
    #     # 'autorange': False,
    #     # 'exponentformat': 'e',
    #     # 'tickformat': ".2f"
    # }

    # yaxis_config = {
    #     'range': [0, 1],
    #     # 'type': 'linear',  # or 'log'
    #     # 'autorange': False,
    #     # 'scaleanchor': 'x',
    #     # 'scaleratio': 1,
    #     # 'exponentformat': 'E',
    #     # 'tickformat': ".2f"
    # }

    # Plot time series using plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[initial_slice_position:initial_slice_position+slice_size], y=time_series, mode='lines', line=dict(color='#636EFA')))
    fig.update_layout(title=f'Sliced close price | Initial position: {initial_slice_position} | Slice size: {slice_size}', 
                      xaxis_title='Date', 
                      yaxis_title='Price',
                    #   xaxis=xaxis_config,
                    #   yaxis=yaxis_config, 
                      width=500,
                      height=500)

    # Save plot as png
    if save:
        fig.write_image(os.path.join(path_to_save, filename + '.png'))
    
    return fig

def generate_plot_full_series_with_highlight(df, initial_slice_position, slice_size, save, path_to_save, filename, data_name):

    fig = go.Figure()
    
    # Plot raw data
    fig.add_trace(go.Scatter(x=df.index, y=df.values, mode='lines', name='Raw Data', line=dict(color='#636EFA')))

    # Highlight a specific index range
    highlight_start_idx = initial_slice_position
    highlight_end_idx = initial_slice_position + slice_size

    # Convert the highlight_start_idx and highlight_end_idx to the corresponding date
    highlight_start_date = df.index[highlight_start_idx]
    highlight_end_date = df.index[highlight_end_idx]

    # Adding rectangular shape to highlight the window
    fig.add_shape(go.layout.Shape(type="rect",xref="x",yref="paper",x0=highlight_start_date,x1=highlight_end_date,y0=0,y1=1,fillcolor="LightSalmon",opacity=0.5,layer="below",line_width=0))

    fig.add_shape(type="line", x0=highlight_end_date, y0=0, x1=highlight_end_date, y1=1,line=dict(color="black",width=2))

    fig.update_layout(title=f'{data_name} | Slice size: {slice_size} | Range: from {highlight_start_date} to {highlight_end_date}', 
                    xaxis_title='Date', 
                    yaxis_title='Price',
                    width=1000,
                    height=500)

    # Save plot as png
    if save:
        fig.write_image(os.path.join(path_to_save, filename + '.png'))
    
    return fig


def generate_features_plot(data, column_name, df_features, slice_position, slice_size, save, path_to_save, filename):

    df = data.copy()
    # get from a certain row to the end based on the index in the df
    df.reset_index(inplace=True)
    df = df[df.index >= slice_size]

    # Convert the highlight_start_idx and highlight_end_idx to the corresponding date
    highlight_start_date = df_features[df_features['initial_slice_position'] == slice_position]['end_date'].values[0]
    highlight_end_date = df_features[df_features['initial_slice_position'] == slice_position]['end_date'].values[0]

    fig = make_subplots(rows=9, cols=1, subplot_titles=('Entropy: Connected components [0]', 'Entropy: Loops [1]', 'Entropy: Voids [2]', 'Amplitude: Connected components [0]', 'Amplitude: Loops [1]', 'Amplitude: Voids [2]', 'Number of points: Connected components [0]', 'Number of points: Loops [1]', 'Number of points: Voids [2]'), specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=df_features['end_date'], y=df_features['connected_components_entropy'], name='Entropy: Connected components [H0]', line=dict(color='#EF553B')), row=1, col=1, secondary_y=False)
    fig.add_shape(type="line", x0=highlight_start_date, y0=0, x1=highlight_end_date, y1=1,line=dict(color="black",width=2),row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=df['date'], y=df[column_name], name=column_name, line=dict(color='#636EFA')),row=1, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(x=df_features['end_date'], y=df_features['loops_entropy'], name='Entropy: Loops [H1]', line=dict(color='#00CC96')), row=2, col=1, secondary_y=False)
    fig.add_shape(type="line", x0=highlight_start_date, y0=0, x1=highlight_end_date, y1=1,line=dict(color="black",width=2),row=2, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=df['date'], y=df[column_name], name=column_name, line=dict(color='#636EFA')),row=2, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(x=df_features['end_date'], y=df_features['voids_entropy'], name='Entropy: Voids [H2]', line=dict(color='#AB63FA')), row=3, col=1, secondary_y=False)
    fig.add_shape(type="line", x0=highlight_start_date, y0=0, x1=highlight_end_date, y1=1,line=dict(color="black",width=2),row=3, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=df['date'], y=df[column_name], name=column_name, line=dict(color='#636EFA')),row=3, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(x=df_features['end_date'], y=df_features['connected_components_amplitude'], name='Amplitude: Connected components [H0]', line=dict(color='#EF553B')), row=4, col=1, secondary_y=False)
    fig.add_shape(type="line", x0=highlight_start_date, y0=0, x1=highlight_end_date, y1=1,line=dict(color="black",width=2),row=4, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=df['date'], y=df[column_name], name=column_name, line=dict(color='#636EFA')),row=4, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(x=df_features['end_date'], y=df_features['loops_amplitude'], name='Amplitude: Loops [H1]', line=dict(color='#00CC96')), row=5, col=1, secondary_y=False)
    fig.add_shape(type="line", x0=highlight_start_date, y0=0, x1=highlight_end_date, y1=1,line=dict(color="black",width=2),row=5, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=df['date'], y=df[column_name], name=column_name, line=dict(color='#636EFA')),row=5, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(x=df_features['end_date'], y=df_features['voids_amplitude'], name='Amplitude: Voids [H2]', line=dict(color='#AB63FA')), row=6, col=1, secondary_y=False)
    fig.add_shape(type="line", x0=highlight_start_date, y0=0, x1=highlight_end_date, y1=1,line=dict(color="black",width=2),row=6, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=df['date'], y=df[column_name], name=column_name, line=dict(color='#636EFA')),row=6, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(x=df_features['end_date'], y=df_features['connected_components_number_of_points'], name='Number of points: Connected components [H0]', line=dict(color='#EF553B')), row=7, col=1, secondary_y=False)
    fig.add_shape(type="line", x0=highlight_start_date, y0=0, x1=highlight_end_date, y1=1,line=dict(color="black",width=2),row=7, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=df['date'], y=df[column_name], name=column_name, line=dict(color='#636EFA')),row=7, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(x=df_features['end_date'], y=df_features['loops_number_of_points'], name='Number of points: Loops [H1]', line=dict(color='#00CC96')), row=8, col=1, secondary_y=False)
    fig.add_shape(type="line", x0=highlight_start_date, y0=0, x1=highlight_end_date, y1=1,line=dict(color="black",width=2),row=8, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=df['date'], y=df[column_name], name=column_name, line=dict(color='#636EFA')),row=8, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(x=df_features['end_date'], y=df_features['voids_number_of_points'], name='Number of points: Voids [H2]', line=dict(color='#AB63FA')), row=9, col=1, secondary_y=False)
    fig.add_shape(type="line", x0=highlight_start_date, y0=0, x1=highlight_end_date, y1=1,line=dict(color="black",width=2),row=9, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=df['date'], y=df[column_name], name=column_name, line=dict(color='#636EFA')),row=9, col=1, secondary_y=True)

    # Remove legend
    fig.update_layout(showlegend=False)

    fig.update_layout(height=1500, width=1000, title_text=f'Sliced {column_name} price | Reference Date: {highlight_start_date} | Slice size: {slice_size}')

        # Save plot as png
    if save:
        fig.write_image(os.path.join(path_to_save, filename + '.png'))
    
    return fig

def normalize_dataframe(df, list_columns_to_normalize, feature_range=(0, 1)):

    # Assume df is your dataframe and cols_to_normalize is the list of columns to normalize
    df_norm = df.copy()

    # Min-Max scaling
    scaler = MinMaxScaler(feature_range=feature_range)
    df_norm[list_columns_to_normalize] = scaler.fit_transform(df[list_columns_to_normalize])

    return df_norm

def accumulate_dataframe(df, list_columns_to_accumulate):

    df_acc = df.copy()

    for column in list_columns_to_accumulate:
        df_acc[column] = df[column].cumsum()

    return df_acc