#%%
import pandas as pd
import streamlit as st
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import s3fs
import os

import nrrd
from PIL import Image, ImageColor
import streamlit.components.v1 as components
from streamlit_plotly_events import plotly_events
import streamlit_nested_layout

import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import image_array_to_data_uri


from streamlit_util import *
# from pipeline import experiment, ephys, lab, psth_foraging, report, foraging_analysis
# from pipeline.plot import foraging_model_plot

CCF_RESOLUTION = 25


cache_folder = '/Users/han.hou/s3-drive/st_cache/'
cache_fig_folder = '/Users/han.hou/Library/CloudStorage/OneDrive-AllenInstitute/pipeline_report/report/all_units/'  # 

if os.path.exists(cache_folder):
    use_s3 = False
else:
    cache_folder = 'aind-behavior-data/Han/ephys/report/st_cache/'
    cache_fig_folder = 'aind-behavior-data/Han/ephys/report/all_units/'
    
    fs = s3fs.S3FileSystem(anon=False)
    use_s3 = True

st.set_page_config(layout="wide", page_title='Foraging unit navigator')

if 'selected_points' not in st.session_state:
    st.session_state['selected_points'] = []

@st.experimental_memo(ttl=24*3600)
def get_fig_unit_all_in_one(key):
    sess_date_str = datetime.strftime(datetime.strptime(key['session_date'], '%Y-%m-%dT%H:%M:%S'), '%Y%m%d')
    
    fn = f'*{key["h2o"]}_{sess_date_str}_{key["ins"]}*u{key["unit"]:03}*'
    aoi = key["area_of_interest"]
    
    if use_s3:
        file = fs.glob(cache_fig_folder + ('' if aoi == 'others' else aoi + '/') + fn)
        if len(file) == 1:
            with fs.open(file[0]) as f:
                img = Image.open(f)
                img = img.crop((500, 140, 5400, 3000))           
    else:
        file = glob.glob(cache_fig_folder + ('' if aoi == 'others' else aoi + '/') + fn)
        if len(file) == 1:
            img = Image.open(file[0])
            img = img.crop((500, 140, 5400, 3000))
            
    return img

# table_mapping = {
#     'sessions': fetch_sessions,
#     'ephys_units': fetch_ephys_units,
# }

@st.experimental_memo(ttl=24*3600)
def load_data(tables=['sessions']):
    df = {}
    for table in tables:
        file_name = cache_folder + f'{table}.pkl'
        if use_s3:
            with fs.open(file_name) as f:
                df[table] = pd.read_pickle(f)
        else:
            df[table] = pd.read_pickle(file_name)
        
    return df
    

df = load_data(['sessions', 'ephys_units'])


@st.experimental_memo(ttl=24*3600)
def get_fig(key):
    fig = plt.figure(figsize=(8, 3), constrained_layout=True)
    ax = fig.subplots(1,1)
    
    foraging_model_plot.plot_session_fitted_choice(key, ax=ax, remove_ignored=False, first_n=2)
    return fig   


@st.experimental_memo(ttl=24*3600)
def plot_scatter(data, x_name='dQ_iti', y_name='sumQ_iti'):
    fig = px.scatter(data, x=x_name, y=y_name, 
                     color='area_of_interest', symbol="area_of_interest",
                     hover_data=['annotation'])
    
    fig.add_vline(x=2.0, line_width=1, line_dash="dash", line_color="black")
    fig.add_vline(x=-2.0, line_width=1, line_dash="dash", line_color="black")
    fig.add_hline(y=2.0, line_width=1, line_dash="dash", line_color="black")
    fig.add_hline(y=-2.0, line_width=1, line_dash="dash", line_color="black")

    fig.add_hline(y=2.0, line_width=1, line_dash="dash", line_color="black")
        
    # fig.update_xaxes(range=[-40, 40])
    # fig.update_yaxes(range=[-40, 40])
    
    # fig.update_layout(width = 800, height = 800)
    
    if all(any([s in name for s in ['dQ', 'sumQ', 'rpe']]) for name in [x_name, y_name]):
        fig.update_yaxes(
            scaleanchor = "x",
            scaleratio = 1,
        )
            
    return fig

@st.experimental_memo(ttl=24*3600)
def load_ccf():
    nrrd_file = f'./data/annotation_{CCF_RESOLUTION}.nrrd'
    color_file = './data/hexcode.csv'
    region_file = './data/mousebrainontology_2.csv'
    
    stack, hdr = nrrd.read(nrrd_file)  
    hexcode = pd.read_csv(color_file, header=None, index_col=0)
    regions = pd.read_csv(region_file, header=None, index_col=0)
  
    hexcode.loc[0] = 'FFFFFF' # Add white
    regions.loc[0] = ''
    
    return stack, hdr, hexcode, regions
   

@st.experimental_memo(ttl=24*3600, persist='disk')
def get_coronal_slice(ccf_x):
    
    stack, _, hexcode, regions = load_ccf()
    ccf_x_ind = round(ccf_x / CCF_RESOLUTION)
    coronal_slice = stack[ccf_x_ind, :, :]
    
    coronal_slice_color = np.full((coronal_slice.shape[0], coronal_slice.shape[1], 3), np.nan)
    coronal_slice_name = np.full((coronal_slice.shape[0], coronal_slice.shape[1], 1), '', dtype=object)
    
    for area_code in np.unique(coronal_slice):
        matched_ind = np.where(coronal_slice == area_code)
        dv_ind, lr_ind = matched_ind
        
        color_code = hexcode.loc[area_code, 1]
        if color_code == '19399':  color_code = '038da6' # Bug fix of missing color
        c_rgb = ImageColor.getcolor("#" + color_code, "RGB")

        coronal_slice_color[dv_ind, lr_ind, :] = c_rgb
        coronal_slice_name[dv_ind, lr_ind, 0] = regions.loc[area_code, 1]
        
    return coronal_slice_color, coronal_slice_name

@st.experimental_memo(ttl=24*3600, experimental_allow_widgets=True)
def plot_coronal_slice_unit(ccf_x):

    coronal_slice, coronal_slice_name = get_coronal_slice(ccf_x)

    img_str = image_array_to_data_uri(
            coronal_slice.astype(np.uint8),
            backend='auto',
            compression=1,
            ext='png', 
            )
        
    hovertemplate = "%%{customdata[0]}<br>%s: %%{x}<br>%s: %%{y}<extra></extra>" % (
            "ccf_z (left -> right)",
            "ccf_y (top -> down)",)
    
    traces = go.Image(source=img_str, x0=0, y0=0, dx=CCF_RESOLUTION, dy=CCF_RESOLUTION,
                      hovertemplate=hovertemplate, customdata=coronal_slice_name)
    fig = go.Figure()
    fig.add_trace(traces)
    fig.update_layout(width=2000, height=800)
    
    
    # fig = px.imshow(coronal_slice.astype(np.uint8), x=np.r_[range(coronal_slice.shape[1])] * CCF_RESOLUTION, 
    #                                y=np.r_[range(coronal_slice.shape[0])] * CCF_RESOLUTION,
    #                                width=2000, height=800,
    #                                labels={'x': 'ccf_z (left -> right)', 'y': 'ccf_y (top -> down)'})
    
    st.plotly_chart(fig, use_container_width=True)
    # st.pyplot(fig)
    return


# ------- Layout starts here -------- #

# st.markdown('## Foraging sessions')
# always_refetch = st.checkbox('Always refetch', value=False)
# ephys_only = st.checkbox('Ephys only', value=True)

# col1, col2 = st.columns([1, 1.5], gap='small')
# with col1:
#     if ephys_only:
#         selection = aggrid_interactive_table_session(df=df['sessions'].query('ephys_ins > 0'))
#     else:
#         selection = aggrid_interactive_table_session(df=df['sessions'])
#         # selection_units = aggrid_interactive_table(df=df['ephys_units'])

# with col2:
#     if selection["selected_rows"]:
#         # st.write("You selected:")
#         # st.json(selection["selected_rows"])
#         fig = get_fig(selection["selected_rows"])
        
#         # fig_html = mpld3.fig_to_html(fig)
#         # components.html(fig_html, height=600)
#         st.write(fig)
        

st.markdown('## Unit browser')
st.write('(data fetched from S3)' if use_s3 else '(data fetched from local)')

col3, col4 = st.columns([1, 2], gap='small')
with col3:
    # -- 1. unit dataframe --
    selection_units = aggrid_interactive_table_units(df=df['ephys_units'])
    st.write(f"{len(selection_units['data'])} units filtered")
    
    # -- 2. axes selector -- 
    unit_stats_names = [keys for keys in df['ephys_units'].keys() if any([s in keys for s in ['dQ', 'sumQ', 'rpe', 'ccf']])]
    with st.expander("Select X and Y axes", expanded=False):
        with st.form("axis_selection"):
            col1, col2 = st.columns([1, 1])
            with col1:
                x_name = st.selectbox("x axis", unit_stats_names, index=3)
            with col2:
                y_name = st.selectbox("y axis", unit_stats_names, index=7)
            st.form_submit_button("update axes")
    
    # Writes a component similar to st.write()
    if len(selection_units['data']):
        fig = plot_scatter(selection_units['data'], x_name=x_name, y_name=y_name)
        
        if len(st.session_state.selected_points):
            fig.add_trace(go.Scatter(x=[st.session_state.selected_points[0]['x']], 
                                 y=[st.session_state.selected_points[0]['y']], 
                            mode = 'markers',
                            marker_symbol = 'star',
                            marker_size = 15,
                            name='selected'))
        
        # Select other Plotly events by specifying kwargs
        selected_points = plotly_events(fig, click_event=True, hover_event=False, select_event=False,
                                        override_height=800, override_width=800)
                
with col4:
    with st.expander("Expand to see all-in-one plot for selected unit"):
        if len(st.session_state.selected_points) == 1:  # Priority to select on scatter plot
            key = df['ephys_units'].query(f'{x_name} == {st.session_state.selected_points[0]["x"]} and {y_name} == {st.session_state.selected_points[0]["y"]}')
            if len(key):
                unit_fig = get_fig_unit_all_in_one(dict(key.iloc[0]))
                st.image(unit_fig, output_format='PNG', width=3000)

        elif len(selection_units['selected_rows']) == 1:
            unit_fig = get_fig_unit_all_in_one(selection_units['selected_rows'][0])
            st.image(unit_fig, output_format='PNG', width=3000)
            

    ccf_x = st.slider("CCF_x", min_value=0, max_value=12000, value=5000, step=100)
    plot_coronal_slice_unit(ccf_x)
    
            
if selected_points and selected_points != st.session_state.selected_points:
    st.session_state.selected_points = selected_points
    st.session_state.selected_points
    st.experimental_rerun()