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
import scipy

import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import image_array_to_data_uri


from streamlit_util import *
# from pipeline import experiment, ephys, lab, psth_foraging, report, foraging_analysis
# from pipeline.plot import foraging_model_plot

CCF_RESOLUTION = 25

cache_folder = 'xxx' # '/Users/han.hou/s3-drive/st_cache/'
cache_fig_folder = 'xxx' # '/Users/han.hou/Library/CloudStorage/OneDrive-AllenInstitute/pipeline_report/report/all_units/'  # 

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
   

@st.experimental_memo(ttl=24*3600, persist='disk', show_spinner=False)
def get_slice(direction, ccf_x):
    
    stack, _, hexcode, regions = load_ccf()
    ccf_x_ind = round(ccf_x / CCF_RESOLUTION)
    
    if direction == 'coronal':
        coronal_slice = stack[ccf_x_ind, :, :]
    else:
        coronal_slice = stack[:, :, ccf_x_ind].swapaxes(0, 1)
    
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
    
    # Edge of areas
    gradient = np.gradient(coronal_slice)
    coronal_edges = np.nonzero((gradient[0] != 0) + (gradient[1] != 0))

    return coronal_slice_color, coronal_slice_name, coronal_edges

@st.experimental_memo(ttl=24*3600)
def _get_min_max():
    x_gamma_all = df['ephys_units'][size_to_map] ** size_gamma
    return np.percentile(x_gamma_all, 5), np.percentile(x_gamma_all, 95)

def _size_mapping(x):
    x_gamma = x**size_gamma
    min_x, max_x = _get_min_max()
    return size_range[0] + x_gamma / (max_x - min_x) * (size_range[1] - size_range[0])

def _smooth_heatmap(data, sigma):
    '''https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python/36307291#36307291'''
    
    U = data.copy()
    V=U.copy()
    V[np.isnan(U)]=0
    VV=scipy.ndimage.gaussian_filter(V,sigma=sigma)

    W=0*U.copy()+1
    W[np.isnan(U)]=0
    WW=scipy.ndimage.gaussian_filter(W,sigma=sigma)

    np.seterr(divide='ignore', invalid='ignore')

    Z=VV/WW
    
    for i in range(len(Z)):
        for j in range(len(Z[0])):
            if np.isnan(U[i][j]):
                Z[i][j] = np.nan
    return Z

# @st.experimental_memo(ttl=24*3600, experimental_allow_widgets=True, show_spinner=False)
def plot_coronal_slice_unit(ccf_x, slice_thickness, if_flip, *args):

    # -- ccf annotation --
    coronal_slice, coronal_slice_name, coronal_edges = get_slice('coronal', ccf_x)
    
    if if_flip:
        max_x = int(np.ceil(5700 / CCF_RESOLUTION))
        coronal_slice = coronal_slice[:, :max_x, :]
        coronal_slice_name = coronal_slice_name[:, :max_x, :]
        coronal_edges = [coord[coronal_edges[1] < max_x] for coord in coronal_edges]

    img_str = image_array_to_data_uri(
            coronal_slice.astype(np.uint8),
            backend='auto',
            compression=0,
            ext='png',
            )
        
    hovertemplate = "%{customdata[0]}<extra></extra>"
            # <br>%s: %%{x}<br>%s: %%{y}<extra></extra>" % (
            # "ccf_z (left -> right)",
            # "ccf_y (top -> down)",)
    
    traces = go.Image(source=img_str, x0=0, y0=0, dx=CCF_RESOLUTION, dy=CCF_RESOLUTION,
                      hovertemplate=hovertemplate, customdata=coronal_slice_name)
    fig = go.Figure()
    fig.add_trace(traces)
    fig.add_trace(go.Scatter(
        x=[300, 300],
        y=[0, 300],
        mode='text',
        text=[f'AP ~ {ccf_x_to_AP(ccf_x)} mm', f'Slice thickness = {slice_thickness} um'],
        textfont=dict(size=20),
        textposition='bottom right',
        showlegend=False
    ))
    
    # -- overlay edges
    xx, yy = coronal_edges
    fig.add_trace(go.Scatter(x=yy * 25, y=xx * 25, 
                             mode='markers',
                             marker={'color': 'rgba(0, 0, 0, 0.5)', 'size': 1},
                             hoverinfo='skip',
                             showlegend=False,
                             ))

    # fig = px.imshow(coronal_slice.astype(np.uint8), x=np.r_[range(coronal_slice.shape[1])] * CCF_RESOLUTION, 
    #                                y=np.r_[range(coronal_slice.shape[0])] * CCF_RESOLUTION,
    #                                width=2000, height=800,
    #                                labels={'x': 'ccf_z (left -> right)', 'y': 'ccf_y (top -> down)'})
    
    # -- overlayed units --
    if len(aggrid_outputs['data']):
        units_to_overlay = aggrid_outputs['data'].query(f'{ccf_x - slice_thickness/2} < ccf_x and ccf_x <= {ccf_x + slice_thickness/2}')
        
        x = units_to_overlay['ccf_z']
        y = units_to_overlay['ccf_y']
        if if_flip:
            x[x > 5700] = 5700 * 2 - x[x > 5700]
            
        if if_ccf_plot_heatmap:
            tuning_strength_2d = scipy.stats.binned_statistic_2d(x=x, y=y, 
                                            values=units_to_overlay[size_to_map], 
                                            statistic=heatmap_aggr_func[0], 
                                            bins=[np.arange(x.min(), x.max(), heatmap_bin_size),
                                                  np.arange(y.min(), y.max(), heatmap_bin_size)])
            
            Z = _smooth_heatmap(tuning_strength_2d.statistic.T, sigma=heatmap_smooth)
            
            fig.add_trace(go.Heatmap(z=Z, 
                                    x=tuning_strength_2d.x_edge, 
                                    y=tuning_strength_2d.y_edge,
                                    zmin=heatmap_color_range[0],
                                    zmax=heatmap_color_range[1],
                                    hoverinfo='skip',
                                    #colorbar=dict(orientation="h", len=1)
                                    ))
    
        
        if if_ccf_plot_scatter:
            x = x + np.random.random(x.shape) * 30

            fig.add_trace(go.Scatter(x=x, 
                                    y=y,
                                    mode = 'markers',
                                    marker_size = _size_mapping(units_to_overlay[size_to_map]),
                                    marker_color = 'black',
                                    hovertemplate= '"%{customdata[0]}"' + 
                                                    '<br>%{text}' +
                                                    '<br>%s = %%{customdata[1]}' % (size_to_map) +
                                                    '<br>uid = %{customdata[2]}' +
                                                    '<extra></extra>',
                                    text=units_to_overlay['annotation'],
                                    customdata=np.stack((units_to_overlay['area_of_interest'], 
                                                        units_to_overlay[size_to_map], 
                                                        units_to_overlay['uid']),
                                                        axis=-1),
                                    showlegend=False
                                    ))
        
    fig.update_layout(width=800 if if_flip else 1000, 
                      height= 1000,
                      xaxis_range=[0, 5700 if if_flip else 5700*2],
                      yaxis_range=[8000, 0],
                      xaxis_title='ccf_z (left -> right)',
                      yaxis_title='ccf_y (superior -> inferior)',
                      )
    
    
    # st.plotly_chart(fig, use_container_width=True)
    # st.pyplot(fig)
    return fig

# @st.experimental_memo(ttl=24*3600, experimental_allow_widgets=True, show_spinner=False)
def plot_saggital_slice_unit(ccf_z, slice_thickness, if_flip, *args):

    # -- ccf annotation --
    slice, slice_name, edges = get_slice('saggital', ccf_z)
    
    # if if_flip:
    #     max_x = int(np.ceil(5700 / CCF_RESOLUTION))
    #     coronal_slice = coronal_slice[:, :max_x, :]
    #     coronal_slice_name = coronal_slice_name[:, :max_x, :]
    #     coronal_edges = [coord[coronal_edges[1] < max_x] for coord in coronal_edges]

    img_str = image_array_to_data_uri(
            slice.astype(np.uint8),
            backend='auto',
            compression=0,
            ext='png',
            )
        
    hovertemplate = "%{customdata[0]}<extra></extra>"
            # <br>%s: %%{x}<br>%s: %%{y}<extra></extra>" % (
            # "ccf_z (left -> right)",
            # "ccf_y (top -> down)",)
    
    traces = go.Image(source=img_str, x0=0, y0=0, dx=CCF_RESOLUTION, dy=CCF_RESOLUTION,
                      hovertemplate=hovertemplate, customdata=slice_name)
    fig = go.Figure()
    fig.add_trace(traces)
    fig.add_trace(go.Scatter(
        x=[300, 300],
        y=[0, 300],
        mode='text',
        text=[f'ML ~ {ccf_z_to_ML(ccf_z)} mm', f'Slice thickness = {slice_thickness} um'],
        textfont=dict(size=20),
        textposition='bottom right',
        showlegend=False
    ))
    
    # -- overlay edges
    xx, yy = edges
    fig.add_trace(go.Scatter(x=yy * 25, y=xx * 25, 
                             mode='markers',
                             marker={'color': 'rgba(0, 0, 0, 0.5)', 'size': 1},
                             hoverinfo='skip',
                             showlegend=False,
                             ))
   
    # -- overlayed units --
    if len(aggrid_outputs['data']):
        aggrid_outputs['data']['ccf_z_in_slice_plot'] = aggrid_outputs['data']['ccf_z']
        if if_flip:
            to_flip_idx = aggrid_outputs['data']['ccf_z_in_slice_plot'] > 5700
            to_flip_value = aggrid_outputs['data']['ccf_z_in_slice_plot'][to_flip_idx]
            aggrid_outputs['data'].loc[to_flip_idx, 'ccf_z_in_slice_plot'] = 2 * 5700 - to_flip_value
            
        units_to_overlay = aggrid_outputs['data'].query(f'{ccf_z - slice_thickness/2} < ccf_z_in_slice_plot and ccf_z_in_slice_plot <= {ccf_z + slice_thickness/2}')
        x = units_to_overlay['ccf_x']
        y = units_to_overlay['ccf_y']
        
        if if_ccf_plot_heatmap:
            tuning_strength_2d = scipy.stats.binned_statistic_2d(x=x, y=y,
                                            values=units_to_overlay[size_to_map], 
                                            statistic=heatmap_aggr_func[0], 
                                            bins=[np.arange(x.min(), x.max(), heatmap_bin_size),
                                                  np.arange(y.min(), y.max(), heatmap_bin_size)])
            
            Z = _smooth_heatmap(tuning_strength_2d.statistic.T, sigma=heatmap_smooth)
            
            fig.add_trace(go.Heatmap(z=Z, 
                                    x=tuning_strength_2d.x_edge, 
                                    y=tuning_strength_2d.y_edge,
                                    zmin=heatmap_color_range[0],
                                    zmax=heatmap_color_range[1],
                                    hoverinfo='skip',
                                    hoverongaps=False,
                                    #colorbar=dict(orientation="h", len=1)
                                    ))
        
        if if_ccf_plot_scatter:
            x = x + np.random.random(x.shape) * 30        
            fig.add_trace(go.Scatter(x=x, 
                                    y=y,
                                    mode = 'markers',
                                    marker_size = _size_mapping(units_to_overlay[size_to_map]),
                                    marker_color = 'black',
                                    hovertemplate= '"%{customdata[0]}"' + 
                                                    '<br>%{text}' +
                                                    '<br>%s = %%{customdata[1]}' % (size_to_map) +
                                                    '<br>uid = %{customdata[2]}' +
                                                    '<extra></extra>',
                                    text=units_to_overlay['annotation'],
                                    customdata=np.stack((units_to_overlay['area_of_interest'], 
                                                        units_to_overlay[size_to_map],
                                                        units_to_overlay['uid']), axis=-1),
                                    showlegend=False
                                    ))
    
    fig.update_layout(width=1600, 
                      height=1000,
                      xaxis_range=[0, 13200],
                      yaxis_range=[8000, 0],
                      xaxis_title='ccf_x (anterior -> posterior)',
                      yaxis_title='ccf_y (superior -> inferior)',
                      )
    
    fig.add_vline(x=ccf_x, line_width=1)
    fig.add_vline(x=max(ccf_x - coronal_thickness/2, fig.layout.xaxis.range[0]), line_width=1, line_dash='dash')
    fig.add_vline(x=min(ccf_x + coronal_thickness/2, fig.layout.xaxis.range[1]), line_width=1, line_dash='dash')
    
    # st.plotly_chart(fig, use_container_width=True)
    # st.pyplot(fig)
    return fig

def ccf_x_to_AP(ccf_x):
    return round(5400 - ccf_x)/1000

def ccf_z_to_ML(ccf_z):
    return round(ccf_z- 5700)/1000

unit_stats_names = [keys for keys in df['ephys_units'].keys() if any([s in keys for s in ['dQ', 'sumQ', 'rpe', 'ccf']])]

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

with st.sidebar:
    with st.expander("CCF view settings", expanded=True):
        size_to_map = st.selectbox("What to plot?", [n for n in unit_stats_names if 'ccf' not in n and 'abs' in n], index=0)
        
        if_flip = st.checkbox("Flip to left hemisphere", value=True)
        
        if_ccf_plot_scatter = st.checkbox("Add units", value=True)
        with st.expander("Unit settings", expanded=True):
            size_range = st.slider("size_range", 0, 50, (0, 10))
            size_gamma = st.slider("gamma", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
        
        if_ccf_plot_heatmap = st.checkbox("Add heatmap", value=True)
        with st.expander("Heatmap settings", expanded=True):
            heatmap_aggr = st.selectbox("Turn which to heatmap?", [r'% significant units', 'median t-stat', 'mean t-stat', 'number of units'], index=0)
            
            sign_level = st.number_input("significant level: t >= ", value=2, disabled=heatmap_aggr != '% significant units')

            heatmap_aggr_func = {'median t-stat': ('median', 0.0, 15.0, 1.0, (2.0, 5.0)),  # func, min, max, step, default
                                'mean t-stat': ('mean', 0.0, 15.0, 1.0, (2.0, 5.0)),
                                r'% significant units': (lambda x: sum(x >= sign_level) / len(x) * 100, 5, 100, 5, (30, 80)),
                                'number of units': (lambda x: len(x) if len(x) else np.nan, 0, 50, 5, (0, 20)),
                                }[heatmap_aggr]
                    
            heatmap_color_range = st.slider(f"Heatmap color range ({heatmap_aggr})", heatmap_aggr_func[1], heatmap_aggr_func[2], step=heatmap_aggr_func[3], value=heatmap_aggr_func[4])
            
            heatmap_bin_size = st.slider("Heatmap bin size", 25, 500, step=25, value=150)
            heatmap_smooth = st.slider("Heatmap smooth factor", 0.0, 2.0, step=0.1, value=1.0)
            

        
with st.container():
    col1, col2 = st.columns([1.5, 1], gap='small')
    with col1:
        # -- 1. unit dataframe --
        aggrid_outputs = aggrid_interactive_table_units(df=df['ephys_units'])
        st.write(f"{len(aggrid_outputs['data'])} units filtered")
        
        # -- axes selector --
        with st.columns([2, 1])[1]:
            with st.expander("Select X and Y axes", expanded=True):
                with st.form("axis_selection"):
                    col3, col4 = st.columns([1, 1])
                    with col3:
                        x_name = st.selectbox("x axis", unit_stats_names, index=3)
                    with col4:
                        y_name = st.selectbox("y axis", unit_stats_names, index=7)
                    st.form_submit_button("update axes")
        
    with col2:
        # -- scatter plot --
        container_scatter = st.container()

with st.container():
    # --- coronal slice ---
    col_coronal, col_saggital = st.columns((1, 1.8))
    with col_coronal:
        ccf_z = st.slider("Saggital slice at (ccf_z)", min_value=600, max_value=5700 if if_flip else 10800, value=5100, step=100) # whole ccf @ 25um [528x320x456] 
        saggital_thickness = st.slider("Slice thickness (LR)", min_value= 100, max_value=5000, step=50, value=700)
        
        container_coronal = st.container()
        
    # --- saggital slice ---
    with col_saggital:
        ccf_x = st.slider("Coronal slice at (ccf_x)", min_value=0, max_value=13100, value=3500, step=100) # whole ccf @ 25um [528x320x456] 
        # st.markdown(f'##### AP relative to Bregma ~ {ccf_x_to_AP(ccf_x): .2f} mm') 
        coronal_thickness = st.slider("Slice thickness (AP)", min_value= 100, max_value=5000, step=50, value=700)
        
        container_saggital = st.container()

container_unit_all_in_one = st.container()

with container_scatter:
    # -- scatter --
    if len(aggrid_outputs['data']):
        fig = plot_scatter(aggrid_outputs['data'], x_name=x_name, y_name=y_name)
        
        if len(st.session_state.selected_points):
            fig.add_trace(go.Scatter(x=[st.session_state.selected_points[0]['x']], 
                                y=[st.session_state.selected_points[0]['y']], 
                            mode = 'markers',
                            marker_symbol = 'star',
                            marker_size = 15,
                            name='selected'))
        
        # Select other Plotly events by specifying kwargs
        selected_points_scatter = plotly_events(fig, click_event=True, hover_event=False, select_event=False,
                                        override_height=800, override_width=800)
 
with container_coronal:
    fig = plot_coronal_slice_unit(ccf_x, coronal_thickness, if_flip, [size_to_map, size_gamma, size_range], aggrid_outputs, ccf_z, saggital_thickness)
    fig.add_vline(x=ccf_z, line_width=1)
    fig.add_vline(x=max(ccf_z - saggital_thickness/2, fig.layout.xaxis.range[0]), line_width=1, line_dash='dash')
    fig.add_vline(x=min(ccf_z + saggital_thickness/2, fig.layout.xaxis.range[1]), line_width=1, line_dash='dash')

    selected_points_slice = plotly_events(fig, click_event=True, hover_event=False, select_event=False,
                                        override_height=1500)
    st.write(selected_points_slice)
    # st.plotly_chart(fig)

with container_saggital:
    fig = plot_saggital_slice_unit(ccf_z, saggital_thickness, if_flip, [size_to_map, size_gamma, size_range], aggrid_outputs, ccf_x, coronal_thickness)
    selected_points_slice = plotly_events(fig, click_event=True, hover_event=False, select_event=False,
                                            override_height=1500)

with container_unit_all_in_one:
    # with st.expander("Expand to see all-in-one plot for selected unit", expanded=True):
    if len(st.session_state.selected_points) == 1:  # Priority to select on scatter plot
        key = df['ephys_units'].query(f'{x_name} == {st.session_state.selected_points[0]["x"]} and {y_name} == {st.session_state.selected_points[0]["y"]}')
        if len(key):
            unit_fig = get_fig_unit_all_in_one(dict(key.iloc[0]))
            st.image(unit_fig, output_format='PNG', width=3000)

    elif len(aggrid_outputs['selected_rows']) == 1:
        unit_fig = get_fig_unit_all_in_one(aggrid_outputs['selected_rows'][0])
        st.image(unit_fig, output_format='PNG', width=3000)

            
if selected_points_scatter and selected_points_scatter != st.session_state.selected_points:
    st.session_state.selected_points = selected_points_scatter
    st.experimental_rerun()