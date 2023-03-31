import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from streamlit_util import filter_dataframe, aggrid_interactive_table_units
from datetime import datetime 

import importlib
package_aoi = importlib.import_module('.2_Area_of_interest_view', package='pages')


import s3fs
from PIL import Image, ImageColor

from Home import add_unit_filter, init


cache_fig_drift_metrics_folder = 'aind-behavior-data/Han/ephys/report/unit_drift_metrics/'
cache_fig_psth_folder = 'aind-behavior-data/Han/ephys/report/all_units/'
fs = s3fs.S3FileSystem(anon=False)

if 'df' not in st.session_state: 
    init()

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


def get_fig_unit_psth_only(key):
    sess_date_str = datetime.strftime(datetime.strptime(key['session_date'], '%Y-%m-%dT%H:%M:%S'), '%Y%m%d')
    
    fn = f'*{key["h2o"]}_{sess_date_str}_{key["ins"]}*u{key["unit"]:03}*'
    aoi = key["area_of_interest"]
    
    file = fs.glob(cache_fig_psth_folder + ('' if aoi == 'others' else aoi + '/') + fn)
    if len(file) == 1:
        with fs.open(file[0]) as f:
            img = Image.open(f)
            img = img.crop((500, 140, 3000, 2800)) 
    else:
        img = None
            
    return img


def get_fig_unit_drift_metric(key):
    fn = f'*{key["subject_id"]}_{key["session"]}_{key["ins"]}_{key["unit"]:03}*'
    
    file = fs.glob(cache_fig_drift_metrics_folder + fn)
    if len(file) == 1:
        with fs.open(file[0]) as f:
            img = Image.open(f)
            img = img.crop((0, 0, img.size[0], img.size[1]))  
    else:
        img = None
            
    return img

draw_func_mapping = {'psth': get_fig_unit_psth_only,
                     'drift metrics': get_fig_unit_drift_metric}

def draw_selected_units(selected_points, draw_types, x_name, y_name, x_abs, y_abs):
    st.write(f'Draw unit plot for selected {len(selected_points)} units')
    my_bar = st.columns((1, 7))[0].progress(0)

    cols = st.columns((1, 1, 1))
    
    for i, xy in enumerate(selected_points):
        q_x = f'abs({x_name}) == {xy["x"]}' if x_abs else f'{x_name} == {xy["x"]}'
        q_y = f'abs({y_name}) == {xy["y"]}' if y_abs else f'{y_name} == {xy["y"]}'
        key = st.session_state.df_unit_filtered.query(f'{q_x} and {q_y}')
        if len(key):
            for draw_type in draw_types:
                img = draw_func_mapping[draw_type](key.iloc[0])
                if img is None:
                    cols[i % 3].markdown(f'{draw_type} fetch error')
                else:
                    cols[i % 3].image(img, output_format='PNG', use_column_width=True)
        else:
            cols[i % 3].markdown('Unit not found')
            
        cols[i % 3].markdown("---")
        my_bar.progress(int((i + 1) / len(selected_points) * 100))
    pass


def add_xy_selector():
    col3, col4 = st.columns([1, 1])
    with col3:
        x_name = st.selectbox("x axis", st.session_state.scatter_stats_names, index=st.session_state.scatter_stats_names.index('poisson_p_choice_outcome'))
        x_abs = st.checkbox(label='abs(x)?')
    with col4:
        y_name = st.selectbox("y axis", st.session_state.scatter_stats_names, index=st.session_state.scatter_stats_names.index('t_dQ_iti'))
        y_abs = st.checkbox(label='abs(y)?')
    return x_name, y_name, x_abs, y_abs

# --------------------------------

with st.sidebar:
    add_unit_filter()
    st.session_state.sign_level = st.number_input("significant level: t >= ", 
                                                value=st.session_state.sign_level if 'sign_level' in st.session_state else 2.57, 
                                                disabled=False, step=1.0) #'significant' not in heatmap_aggr_name, step=1.0)


st.session_state.aggrid_outputs = aggrid_interactive_table_units(df=st.session_state.df_unit_filtered)
col1, col2 = st.columns((1, 1))

with col2:
    x_name, y_name, x_abs, y_abs = add_xy_selector()
    draw_types = st.multiselect('Which plot(s) to draw?', ['psth', 'drift metrics'], default=['psth'])
with col1:
    selected_points_scatter = package_aoi.add_scatter_return_selected(st.session_state.df_unit_filtered, x_name, y_name, x_abs, y_abs)

# fig = plot_drift_metric_scatter(st.session_state.df_unit_filtered)
# selected_points_scatter = plotly_events(fig, click_event=True, hover_event=False, select_event=True, 
#                                         override_height=800, override_width=800)


draw_selected_units(selected_points_scatter, draw_types, x_name, y_name, x_abs, y_abs)