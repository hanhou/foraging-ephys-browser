import plotly.graph_objects as go
import numpy as np
from PIL import Image, ImageColor
import pandas as pd
import scipy
from plotly.utils import image_array_to_data_uri

import streamlit as st
from streamlit_plotly_events import plotly_events
from streamlit_util import *

from Home import add_unit_filter

import nrrd


CCF_RESOLUTION = 25

@st.experimental_memo(ttl=24*3600)
def _get_min_max():
    x_gamma_all = np.abs(st.session_state.df['ephys_units'][value_to_map] ** size_gamma)
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



# @st.experimental_memo(ttl=24*3600, experimental_allow_widgets=True, show_spinner=True)
def draw_ccf_annotations(fig, slice, slice_name, edges, message):
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
    fig.add_trace(traces)
    fig.add_trace(go.Scatter(
        x=[300, 300],
        y=[0, 300],
        mode='text',
        text=message,
        textfont=dict(size=20),
        textposition='bottom right',
        showlegend=False
    ))
    
    # -- overlay edges
    xx, yy = edges
    fig.add_trace(go.Scatter(x=yy * CCF_RESOLUTION, y=xx * CCF_RESOLUTION, 
                             mode='markers',
                             marker={'color': 'rgba(0, 0, 0, 0.3)', 'size': 2},
                             hoverinfo='skip',
                             showlegend=False,
                             ))
    return fig



@st.experimental_memo(persist='disk', show_spinner=False)
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
   

def draw_ccf_heatmap(fig, x, y, z, slice_name):
    
    heatmap = scipy.stats.binned_statistic_2d(x=x, y=y,
                                values=z, 
                                statistic=heatmap_aggr_func, 
                                bins=[np.arange(x.min() - heatmap_bin_size, x.max() + heatmap_bin_size, heatmap_bin_size),
                                      np.arange(y.min() - heatmap_bin_size, y.max() + heatmap_bin_size, heatmap_bin_size)])
    
    heatmap_smoothed = _smooth_heatmap(heatmap.statistic, sigma=heatmap_smooth)
    
    # Build region name
    heatmap_region = np.full(heatmap_smoothed.shape, '', dtype=object)
    ind_xs, ind_ys = np.where(~np.isnan(heatmap_smoothed))
    for ind_x, ind_y in zip(ind_xs, ind_ys):
        ind_x_ccf = int(np.round(heatmap.x_edge[ind_x] / CCF_RESOLUTION))
        ind_y_ccf = int(np.round(heatmap.y_edge[ind_y] / CCF_RESOLUTION))
        heatmap_region[ind_x, ind_y] = (slice_name[ind_y_ccf, ind_x_ccf])
        
    zmin, zmax =(-heatmap_color_range, heatmap_color_range) if if_bi_directional_heatmap else (heatmap_color_range[0], heatmap_color_range[1])
        
    fig.add_trace(go.Heatmap(z=heatmap_smoothed.T, 
                            x=heatmap.x_edge, 
                            y=heatmap.y_edge,
                            zmin=zmin,
                            zmax=zmax,
                            hoverongaps=False,
                            hovertemplate = "%{customdata[0]}" + 
                                            "<br>%s (%s): %%{z}" % (heatmap_aggr_name, value_to_map) + 
                                            "<extra></extra>",
                            customdata=heatmap_region.T,
                            colorscale='RdBu' if if_bi_directional_heatmap else None, 
                            colorbar=dict(orientation="h", len=0.3)
                            ))
    
    return

def draw_ccf_units(fig, x, y, z, aoi, uid, annot):
    x = x + np.random.random(x.shape) * 30    
    if sum(z < 0) == 0:   # All positive values
        fig.add_trace(go.Scatter(x=x, 
                                y=y,
                                mode = 'markers',
                                marker_size = _size_mapping(z),
                                # continuous_color_scale = 'RdBu',
                                marker = {'color': 'rgba(0, 0, 0, 0.8)'},
                                hovertemplate= '"%{customdata[0]}"' + 
                                                '<br>%{text}' +
                                                '<br>%s = %%{customdata[1]}' % value_to_map +
                                                '<br>uid = %{customdata[2]}' +
                                                '<extra></extra>',
                                text=annot,
                                customdata=np.stack((aoi, z, uid), axis=-1),
                                showlegend=False
                                ))
    else:  # Negative: red (for unit_stats: ipsi), positive: blue (for unit_stats: contra)
        for select_z, col in zip((z < 0, z >= 0), ('rgba(255, 0, 0, 0.8)', 'rgba(0, 0, 255, 0.8)')):
            fig.add_trace(go.Scatter(x=x[select_z], 
                                    y=y[select_z],
                                    mode = 'markers',
                                    marker_size = _size_mapping(abs(z[select_z])),
                                    # continuous_color_scale = 'RdBu',
                                    marker = {'color': col},
                                    hovertemplate= '"%{customdata[0]}"' + 
                                                    '<br>%{text}' +
                                                    '<br>%s = %%{customdata[1]}' % value_to_map +
                                                    '<br>uid = %{customdata[2]}' +
                                                    '<extra></extra>',
                                    text=annot[select_z],
                                    customdata=np.stack((aoi[select_z], z[select_z], uid[select_z]), axis=-1),
                                    showlegend=False
                                    ))
        
    return


# @st.experimental_memo(ttl=24*3600, experimental_allow_widgets=True, show_spinner=True)
def plot_coronal_slice_unit(ccf_x, coronal_slice_thickness, if_flip, *args):
    fig = go.Figure()

    # -- ccf annotation --
    coronal_slice, coronal_slice_name, coronal_edges = get_slice('coronal', ccf_x)
    
    if if_flip:
        max_x = int(np.ceil(5700 / CCF_RESOLUTION))
        coronal_slice = coronal_slice[:, :max_x, :]
        coronal_slice_name = coronal_slice_name[:, :max_x, :]
        coronal_edges = [coord[coronal_edges[1] < max_x] for coord in coronal_edges]
        
    message = [f'AP ~ {ccf_x_to_AP(ccf_x)} mm', 
               f'Slice thickness = {coronal_slice_thickness} um']
    fig = draw_ccf_annotations(fig, coronal_slice, coronal_slice_name, 
                               coronal_edges, message)

    # -- overlayed units --
    units_to_overlay = st.session_state.df_unit_filtered.query(f'{ccf_x - coronal_slice_thickness/2} < ccf_x and ccf_x <= {ccf_x + coronal_slice_thickness/2}')
    
    if len(units_to_overlay):
    
        x = units_to_overlay['ccf_z']
        y = units_to_overlay['ccf_y']
        z = units_to_overlay[value_to_map]
        
        if if_take_abs:
            z = np.abs(z)

        if if_flip:
            x[x > 5700] = 5700 * 2 - x[x > 5700]
            
        if if_ccf_plot_heatmap:
            try:
                draw_ccf_heatmap(fig, x, y, z, coronal_slice_name)
            except:
                pass
        
        if if_ccf_plot_scatter:
            aoi = units_to_overlay['area_of_interest']
            uid = units_to_overlay['uid']
            annot = units_to_overlay['annotation']
            try:
                draw_ccf_units(fig, x, y, z, aoi, uid, annot)
            except:
                pass
        
    fig.update_layout(width=800 if if_flip else 1000, 
                      height= 1000,
                      xaxis_range=[0, 5700 if if_flip else 5700*2],
                      yaxis_range=[8000, 0],
                      xaxis_title='ccf_z (left -> right)',
                      yaxis_title='ccf_y (superior -> inferior)',
                      font=dict(size=20)
                      )
    
    # st.plotly_chart(fig, use_container_width=True)
    # st.pyplot(fig)
    return fig


# @st.experimental_memo(ttl=24*3600, experimental_allow_widgets=True, show_spinner=True)
def plot_saggital_slice_unit(ccf_z, saggital_slice_thickness, if_flip, *args):
    fig = go.Figure()

    # -- ccf annotation --
    saggital_slice, saggital_slice_name, saggital_edges = get_slice('saggital', ccf_z)

    message = [f'ML ~ {ccf_z_to_ML(ccf_z)} mm', 
               f'Slice thickness = {saggital_slice_thickness} um']

    fig = draw_ccf_annotations(fig, saggital_slice, saggital_slice_name,
                               saggital_edges, message)

    # -- overlayed units --
    if len(st.session_state.df_unit_filtered):
        st.session_state.df_unit_filtered['ccf_z_in_slice_plot'] = st.session_state.df_unit_filtered['ccf_z']
        if if_flip:
            to_flip_idx = st.session_state.df_unit_filtered['ccf_z_in_slice_plot'] > 5700
            to_flip_value = st.session_state.df_unit_filtered['ccf_z_in_slice_plot'][to_flip_idx]
            st.session_state.df_unit_filtered.loc[to_flip_idx, 'ccf_z_in_slice_plot'] = 2 * 5700 - to_flip_value
            
        units_to_overlay = st.session_state.df_unit_filtered.query(f'{ccf_z - saggital_slice_thickness/2} < ccf_z_in_slice_plot and ccf_z_in_slice_plot <= {ccf_z + saggital_slice_thickness/2}')
    
    if len(units_to_overlay):
        x = units_to_overlay['ccf_x']
        y = units_to_overlay['ccf_y']
        z = units_to_overlay[value_to_map]

        if if_take_abs:
            z = np.abs(z)

        if if_ccf_plot_heatmap:
            draw_ccf_heatmap(fig, x, y, z, saggital_slice_name)
        
        if if_ccf_plot_scatter: 
            aoi = units_to_overlay['area_of_interest']
            annot = units_to_overlay['annotation']
            uid = units_to_overlay['uid']
            
            draw_ccf_units(fig, x, y, z, aoi, uid, annot)
    
    fig.update_layout(width=1300, 
                      height=1000,
                      xaxis_range=[0, 10000],
                      yaxis_range=[8000, 0],
                      xaxis_title='ccf_x (anterior -> posterior)',
                      yaxis_title='ccf_y (superior -> inferior)',
                      font=dict(size=20)
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


def _ccf_heatmap_available_aggr_funcs(value_to_map):
    if 'pure' not in value_to_map:
        return ([r'% significant units'] if all(s not in value_to_map for s in ['avg_firing_rate', 'beta']) 
                else []) + ['median', 'median (significant only)', 'mean', 'mean (significant only)', 'number of units']
    else:
        return [r'% pure units']

def _ccf_heatmap_get_aggr_func(heatmap_aggr_name, value_to_map):
    heatmap_aggr_func_mapping = {    # func, (min, max, step, default)
        'median': ('median', (0.0, 15.0, 1.0, 5.0 if if_bi_directional_heatmap else (2.0, 5.0))),  
        'median (significant only)': (lambda x: np.median(x[np.abs(x) >= sign_level]), (0.0, 15.0, 1.0, 5.0 if if_bi_directional_heatmap else (2.0, 5.0))),  
        'mean': ('mean', (0.0, 15.0, 1.0, 5.0 if if_bi_directional_heatmap else (2.0, 5.0))),
        'mean (significant only)': (lambda x: np.mean(x[np.abs(x) >= sign_level]), (0.0, 15.0, 1.0, 5.0 if if_bi_directional_heatmap else (2.0, 5.0))),
        r'% significant units': (lambda x: sum(np.abs(x) >= sign_level) / len(x) * 100, (5, 100, 5, (30, 80))),
        'number of units': (lambda x: len(x) if len(x) else np.nan, (0, 50, 5, (0, 20))),
        r'% pure units': (lambda x: sum(x) / len(x) * 100, (0, 100, 1, (5, 80))),
        # 'max': ('max', (0.0, 15.0, 1.0, (0.0, 5.0))),
        # 'min': ('min', (0.0, 15.0, 1.0, (0.0, 5.0))),
    }
                                
    return heatmap_aggr_func_mapping[heatmap_aggr_name][0], heatmap_aggr_func_mapping[heatmap_aggr_name][1]


with st.sidebar:    
    add_unit_filter()

    with st.expander("CCF view settings", expanded=True):
        
        if_flip = st.checkbox("Flip to left hemisphere", value=True)
        value_to_map = st.selectbox("Plot which?", st.session_state.ccf_stat_names, index=st.session_state.ccf_stat_names.index('t_dQ_iti'))
        if_take_abs = st.checkbox("Use abs()?", value=False)
        
        if_ccf_plot_scatter = st.checkbox("Draw units", value=True)        
        with st.expander("Unit settings", expanded=True):
            size_range = st.slider("size range", 0, 50, (0, 10))
            size_gamma = st.slider("gamma", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
        
        if_ccf_plot_heatmap = st.checkbox("Draw heatmap", value=True)
        with st.expander("Heatmap settings", expanded=True):
            heatmap_aggr_name = st.selectbox("aggregate function", _ccf_heatmap_available_aggr_funcs(value_to_map), index=0)
                        
            if_bi_directional_heatmap = not (if_take_abs or any(s in value_to_map for s in ['rate']) or 'units' in heatmap_aggr_name) # number_or_units or % sign_units
            heatmap_aggr_func, heatmap_color_ranges = _ccf_heatmap_get_aggr_func(heatmap_aggr_name, value_to_map)
            
            heatmap_color_range = st.slider(f"Heatmap color range ({heatmap_aggr_name})", heatmap_color_ranges[0], heatmap_color_ranges[1], step=heatmap_color_ranges[2], value=heatmap_color_ranges[3])
            heatmap_bin_size = st.slider("Heatmap bin size", 25, 500, step=25, value=100)
            heatmap_smooth = st.slider("Heatmap smooth factor", 0.0, 2.0, step=0.1, value=1.0)
            
    sign_level = st.number_input("significant level: t >= ", value=2.57, disabled=False, step=1.0) #'significant' not in heatmap_aggr_name, step=1.0)



# --- coronal slice ---
col_coronal, col_saggital = st.columns((1, 1.8))
with col_coronal:
    ccf_z = st.slider("Saggital slice at (ccf_z)", min_value=600, max_value=5700 if if_flip else 10800, 
                                    value=st.session_state.ccf_z if 'ccf_z' in st.session_state else 5100, 
                                    step=100)       # whole ccf @ 25um [528x320x456] 
    saggital_thickness = st.slider("Slice thickness (LR)", min_value= 100, max_value=5000, step=50, value=700)
    
    container_coronal = st.container()
    
# --- saggital slice ---
with col_saggital:
    ccf_x = st.slider("Coronal slice at (ccf_x)", min_value=0, max_value=13100, value=3500, step=100) # whole ccf @ 25um [528x320x456] 
    # st.markdown(f'##### AP relative to Bregma ~ {ccf_x_to_AP(ccf_x): .2f} mm') 
    coronal_thickness = st.slider("Slice thickness (AP)", min_value= 100, max_value=5000, step=50, value=700)
    
    container_saggital = st.container()

container_unit_all_in_one = st.container()


with container_coronal:
    fig = plot_coronal_slice_unit(ccf_x, coronal_thickness, if_flip) #, [size_to_map, size_gamma, size_range], aggrid_outputs, ccf_z, saggital_thickness)
    fig.add_vline(x=ccf_z, line_width=1)
    fig.add_vline(x=max(ccf_z - saggital_thickness/2, fig.layout.xaxis.range[0]), line_width=1, line_dash='dash')
    fig.add_vline(x=min(ccf_z + saggital_thickness/2, fig.layout.xaxis.range[1]), line_width=1, line_dash='dash')

    selected_points_slice = plotly_events(fig, click_event=True, hover_event=False, select_event=False,
                                        override_height=1500)
    st.write(selected_points_slice)
    # st.plotly_chart(fig, use_container_width=True)

with container_saggital:
    fig = plot_saggital_slice_unit(ccf_z, saggital_thickness, if_flip) #, [size_to_map, size_gamma, size_range], aggrid_outputs, ccf_x, coronal_thickness)
    selected_points_slice = plotly_events(fig, click_event=True, hover_event=False, select_event=False,
                                            override_height=1500)
    # st.plotly_chart(fig, use_container_width=True)
