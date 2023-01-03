import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from streamlit_util import filter_dataframe, aggrid_interactive_table_units
from streamlit_plotly_events import plotly_events

import s3fs
from PIL import Image, ImageColor

from Home import add_unit_filter


cache_fig_folder = 'aind-behavior-data/Han/ephys/report/unit_drift_metrics/'
fs = s3fs.S3FileSystem(anon=False)


def plot_drift_metric_scatter(data):
    fig = go.Figure()

    for aoi in st.session_state.df['aoi'].index:
        if aoi not in data.area_of_interest.values:
            continue
        
        this_aoi = data.query(f'area_of_interest == "{aoi}"')
        fig.add_trace(go.Scatter(x=this_aoi['poisson_p_choice_outcome'], 
                                y=this_aoi['poisson_p_dave'],
                                mode="markers",
                                marker_color=st.session_state.aoi_color_mapping[aoi],
                                name=aoi))
        
        fig.update_layout(
                    height=800,
                    xaxis_title='Drift metric grouped by choice and outcome',
                    yaxis_title='Drift metric all (Dave)',
                    font=dict(size=20)
                    ) 
        
    return fig

def get_fig_unit_drift_metric(key):
    fn = f'*{key["subject_id"]}_{key["session"]}_{key["ins"]}_{key["unit"]:03}*'
    
    file = fs.glob(cache_fig_folder + fn)
    if len(file) == 1:
        with fs.open(file[0]) as f:
            img = Image.open(f)
            img = img.crop((0, 0, img.size[0], img.size[1]))         
            
    return img


def plot_drift_metric_time_course(selected_points):
    st.write(f'Draw unit drift metrics for selected {len(selected_points)} units')
    my_bar = st.columns((1, 7))[0].progress(0)

    cols = st.columns((1, 1, 1))
    
    for i, xy in enumerate(selected_points):
        key = st.session_state.df_unit_filtered.query(f'poisson_p_choice_outcome == {xy["x"]} '
                                                            f'and poisson_p_dave == {xy["y"]}')
        img = get_fig_unit_drift_metric(key.iloc[0])
        cols[i % 3].image(img, output_format='PNG')
        my_bar.progress(int((i + 1) / len(selected_points) * 100))
    pass

with st.sidebar:
    add_unit_filter()


st.session_state.aggrid_outputs = aggrid_interactive_table_units(df=st.session_state.df_unit_filtered)

fig = plot_drift_metric_scatter(st.session_state.df_unit_filtered)
selected_points_scatter = plotly_events(fig, click_event=True, hover_event=False, select_event=True, 
                                        override_height=800, override_width=800)

plot_drift_metric_time_course(selected_points_scatter)
