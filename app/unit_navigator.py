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


if_profile = False

if if_profile:
    from streamlit_profiler import Profiler
    p = Profiler()
    p.start()


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
    
df = load_data(['sessions', 'ephys_units', 'aoi'])
aoi_color_mapping = {area: f'rgb({",".join(col.astype(str))})' for area, col in zip(df['aoi'].index, df['aoi'].rgb)}

pure_unit_color_mapping =  {'pure_dQ': 'darkviolet',
                            'pure_sumQ': 'deepskyblue',
                            'pure_contraQ': 'darkblue',
                            'pure_ipsiQ': 'darkorange'}

sig_prop_color_mapping =  {'dQ': 'darkviolet',
                            'sumQ': 'deepskyblue',
                            'contraQ': 'darkblue',
                            'ipsiQ': 'darkorange',
                            'rpe': 'gray'}


@st.experimental_memo(ttl=24*3600)
def get_fig(key):
    fig = plt.figure(figsize=(8, 3), constrained_layout=True)
    ax = fig.subplots(1,1)
    
    foraging_model_plot.plot_session_fitted_choice(key, ax=ax, remove_ignored=False, first_n=2)
    return fig   


@st.experimental_memo(ttl=24*3600)
def plot_scatter(data, x_name='dQ_iti', y_name='sumQ_iti', if_use_ccf_color=False, sign_level=2.57):
    
    fig = go.Figure()
    
    for aoi in df['aoi'].index:
        if aoi not in df_unit_filtered.area_of_interest.values:
            continue
        
        this_aoi = data.query(f'area_of_interest == "{aoi}"')
        fig.add_trace(go.Scatter(x=this_aoi[x_name], 
                                 y=this_aoi[y_name],
                                 mode="markers",
                                 marker_color=aoi_color_mapping[aoi] if if_use_ccf_color else None,
                                 name=aoi))
        
    # fig = px.scatter(data, x=x_name, y=y_name, 
    #                 color='area_of_interest', symbol="area_of_interest",
    #                 hover_data=['annotation'],
    #                 color_discrete_map=aoi_color_mapping if if_use_ccf_color else None)
    
    if 't_' in x_name:
        fig.add_vline(x=sign_level, line_width=1, line_dash="dash", line_color="black")
        fig.add_vline(x=-sign_level, line_width=1, line_dash="dash", line_color="black")
    if 't_' in y_name:
        fig.add_hline(y=sign_level, line_width=1, line_dash="dash", line_color="black")
        fig.add_hline(y=-sign_level, line_width=1, line_dash="dash", line_color="black")
        
    # fig.update_xaxes(range=[-40, 40])
    # fig.update_yaxes(range=[-40, 40])
    
    fig.update_layout(width=900, height=800, font=dict(size=20), xaxis_title=x_name, yaxis_title=y_name)
    
    if all(any([s in name for s in ['t_', 'beta_']]) for name in [x_name, y_name]):
        fig.update_yaxes(
            scaleanchor = "x",
            scaleratio = 1,
        )
            
    return fig


@st.experimental_memo(ttl=24*3600)
def plot_unit_class_scatter(x_name, y_name):

    fig = go.Figure()
    
    for unit_class, color in pure_unit_color_mapping.items():
        this = df_unit_filtered[f'{unit_class}_dQ_sumQ']
        fig.add_trace(go.Scatter(x=df_unit_filtered[x_name][this], 
                                    y=df_unit_filtered[y_name][this], 
                                    mode='markers',
                                    marker_color=color, 
                                    name=f'{unit_class}_dQ_sumQ'
                                ))
        
    fig.add_trace(go.Scatter(x=df_unit_filtered.query('p_model_iti >= 0.01')[x_name], 
                                y=df_unit_filtered.query('p_model_iti >= 0.01')[y_name], 
                            mode='markers',
                            marker_color='black', 
                            name='non_sig'
                    ))
    fig.update_layout(width=500, height=500, font=dict(size=20))
    fig.update_yaxes(scaleanchor = "x", scaleratio = 1)
            
    return fig

@st.experimental_memo(ttl=24*3600)
def plot_unit_pure_class_bar():
    linear_model = '_dQ_sumQ'
    fig = go.Figure()
    for pure_class, color in pure_unit_color_mapping.items():
        data = df['aoi'][pure_class + linear_model].values
        prop = [x[0] for x in data]
        err = [x[1] for x in data]
        fig.add_trace(go.Bar(
                            name=pure_class,
                            x=df['aoi'].index, 
                            y=prop,
                            error_y=dict(type='data', array=err),
                            marker=dict(color=color),
                            hovertemplate='%%{x}, %s' % (pure_class) + 
                                          '<br>%{y:.1f} ± %{customdata[0]:.1f} % (95% CI)' + 
                                          '<br>n = %{customdata[1]} <extra></extra>',
                            customdata=np.stack((err, df['aoi'].number_of_units), axis=-1),
                            ))
    
    fig.add_hline(y=0.01 * 100 / 4, line_width=1, line_dash="dash", line_color="black", name='Type I error')  # Type I error (I used p < 0.01 when classify units) devided by 4 types
    fig.update_layout(barmode='group', 
                      height=800,
                      yaxis_title='% pure units (+/- 95% CI)',
                      font=dict(size=20)
                      )    
    return fig

# Add proportion of pure units
def _sig_proportion(ts):
    prop = np.sum(np.abs(ts) >= sign_level) / len(ts)
    ci_95 = 1.96 * np.sqrt(prop * (1 - prop) / len(ts))
    return prop * 100, ci_95 * 100, len(ts)

@st.experimental_memo(ttl=24*3600)
def plot_unit_sig_prop_bar(df_unit_filtered, sign_level):
    
    fig = go.Figure()
    period = 'iti'    
        
    for stat_name, color in sig_prop_color_mapping.items():
        col = f't_{stat_name}_{period}'
        
        prop_ci = df_unit_filtered.groupby('area_of_interest')[col].agg(_sig_proportion)
        filtered_aoi = [col for col in df['aoi'].index if col in prop_ci] # Sort according to df['aoi'].index
        prop_ci = prop_ci.reindex(filtered_aoi)
        prop = [x[0] for x in prop_ci.values] 
        err = [x[1] for x in prop_ci.values] 
        ns =  [x[2] for x in prop_ci.values] 

        fig.add_trace(go.Bar( 
                            name=f'{stat_name}, {period}',
                            x=filtered_aoi,
                            y=prop,
                            error_y=dict(type='data', array=err),
                            hovertemplate='%%{x}, %s, %s' % (stat_name, period) + 
                                          '<br>%{y:.1f} ± %{customdata[0]:.1f} % (95% CI)' + 
                                          '<br>n = %{customdata[1]} <extra></extra>',
                            customdata=np.stack((err, ns), axis=-1),
                            marker=dict(
                                        color=color,
                                        line_color=color,
                                        line_width=3,
                                        pattern_shape='/', 
                                        pattern_fillmode="replace",
                                        ),
                            ))
    
    fig.update_layout(barmode='group', 
                      height=800,
                      yaxis_title='% sig. units (+/- 95% CI)',
                      font=dict(size=20)
                      )    
    return fig


@st.experimental_memo(ttl=24*3600)
def _polar_histogram(df_this_aoi, x_name, y_name, polar_method, bins):

    df_sig = df_this_aoi.query(f'abs({x_name}) >= {sign_level} or abs({y_name}) >= {sign_level}')
    theta, r = _to_theta_r(df_sig[x_name], df_sig[y_name])
    weight = r if 'weighted' in polar_method else np.ones_like(theta)

    counts, _ = np.histogram(a=theta, bins=bins, weights=weight)
    
    if 'in all neurons' in polar_method:
        return counts / len(df_this_aoi) * 100 # Not sum to 1
    elif 'in significant neurons' in polar_method:
        return counts / len(df_sig) * 100 # Sum to 1
    else:
        return counts / np.sum(counts)  # weighted r


@st.experimental_memo(ttl=24*3600)        
def plot_polar(df_unit_filtered, x_name, y_name, polar_method, n_bins, if_errorbar):
        
    bins = np.linspace(-np.pi, np.pi, num=n_bins + 1)
    bin_center = np.rad2deg(np.mean([bins[:-1], bins[1:]], axis=0)) 

    polar_hist = df_unit_filtered.groupby('area_of_interest').apply(lambda df_this_aoi: _polar_histogram(df_this_aoi, x_name, y_name, polar_method, bins))

    fig = go.Figure() 
    
    for aoi in df['aoi'].index:
        if aoi not in df_unit_filtered.area_of_interest.values:
            continue
        
        hist = polar_hist[aoi]
        fig.add_trace(go.Scatterpolar(r=np.hstack([hist, hist[0]]),
                                        theta=np.hstack([bin_center, bin_center[0]]),
                                        mode='lines + markers',
                                        marker_color=aoi_color_mapping[aoi], 
                                        legendgroup=aoi,
                                        name=aoi,
                    )
                    #   color_discrete_sequence=px.colors.sequential.Plasma_r,
                    #   template="plotly_dark",
                    )
        
        # add binomial errorbar
        if 'in all neurons' in polar_method and if_errorbar:
            n = len(df_unit_filtered.query(f'area_of_interest == "{aoi}"')) 
            for p, theta in zip(hist, bin_center):
                ci_95 = 1.96 * np.sqrt(p * (100 - p) / n)
                fig.add_trace(go.Scatterpolar(r=[p - ci_95, p + ci_95],
                                              theta=[theta, theta],
                                              mode='lines',
                                              marker_color=aoi_color_mapping[aoi],
                                              legendgroup=aoi,
                                              name=aoi,
                                              showlegend=False,
                                              ))
    
    # add type I error
    fig.add_trace(go.Scatterpolar(r=np.full(len(bin_center) + 1, 0.01 * 100 / n_bins),
                                  theta=np.hstack([bin_center, bin_center[0]]),
                                  mode='lines',
                                  line=dict(color='black', dash='dot'),
                                  name='Type I error'
                                  )) # Type I error divided by 4
    
    fig.update_layout(height=800, width=800, font=dict(size=20))
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
def _get_min_max():
    x_gamma_all = np.abs(df['ephys_units'][value_to_map] ** size_gamma)
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

def _to_theta_r(x, y):
    return np.arctan2(y, x), np.sqrt(x**2 + y**2)
    

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
    units_to_overlay = df_unit_filtered.query(f'{ccf_x - coronal_slice_thickness/2} < ccf_x and ccf_x <= {ccf_x + coronal_slice_thickness/2}')
    
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
    if len(df_unit_filtered):
        df_unit_filtered['ccf_z_in_slice_plot'] = df_unit_filtered['ccf_z']
        if if_flip:
            to_flip_idx = df_unit_filtered['ccf_z_in_slice_plot'] > 5700
            to_flip_value = df_unit_filtered['ccf_z_in_slice_plot'][to_flip_idx]
            df_unit_filtered.loc[to_flip_idx, 'ccf_z_in_slice_plot'] = 2 * 5700 - to_flip_value
            
        units_to_overlay = df_unit_filtered.query(f'{ccf_z - saggital_slice_thickness/2} < ccf_z_in_slice_plot and ccf_z_in_slice_plot <= {ccf_z + saggital_slice_thickness/2}')
    
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

scatter_stats_names = [keys for keys in df['ephys_units'].keys() if any([s in keys for s in ['dQ', 'sumQ', 'contraQ', 'ipsiQ', 'rpe', 'ccf', 'firing_rate']])]
ccf_stat_names = [n for n in scatter_stats_names if 'ccf' not in n]

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


# ------- Layout starts here -------- #    

st.markdown('## Foraging Unit Browser')


with st.sidebar:
                
    with st.expander("CCF view settings", expanded=True):
        
        if_flip = st.checkbox("Flip to left hemisphere", value=True)
        value_to_map = st.selectbox("Plot which?", ccf_stat_names, index=ccf_stat_names.index('t_dQ_iti'))
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

        
with st.container():
    # col1, col2 = st.columns([1.5, 1], gap='small')
    # with col1:
    # -- 1. unit dataframe --
    st.markdown('### Filtering units in this table has global effect (select one line to plot unit below)')
    aggrid_outputs = aggrid_interactive_table_units(df=df['ephys_units'])
    df_unit_filtered = aggrid_outputs['data']
    
    st.write(f"{len(df_unit_filtered)} units filtered" + ' (data fetched from S3)' if use_s3 else '(data fetched from local)')
    st.markdown('### Select views')

tabs_font_css = """
<style>
button[data-baseweb="tab"] {
  font-size: 20px;
}
</style>
"""
st.write(tabs_font_css, unsafe_allow_html=True)
tab_ccf_view, tab_aoi_view = st.tabs(['**1. CCF VIEW**', "**2. AREA OF INTEREST VIEW**"])


with tab_ccf_view:
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


with tab_aoi_view:
    # -- axes selector --
    with st.expander("Select axes", expanded=False):
        with st.form("axis_selection"):
            col3, col4 = st.columns([1, 1])
            with col3:
                x_name = st.selectbox("x axis", scatter_stats_names, index=scatter_stats_names.index('t_dQ_iti'))
            with col4:
                y_name = st.selectbox("y axis", scatter_stats_names, index=scatter_stats_names.index('t_sumQ_iti'))
            st.form_submit_button("update axes")

    # -- scatter --
    col1, col2 = st.columns((1, 1))
    with col1:  # Raw scatter
        if len(df_unit_filtered):
            if_use_ccf_color = st.checkbox("Use ccf color", value=True)
            fig = plot_scatter(df_unit_filtered, x_name=x_name, y_name=y_name, if_use_ccf_color=if_use_ccf_color, sign_level=sign_level)
            
            if len(st.session_state.selected_points):
                fig.add_trace(go.Scatter(x=[st.session_state.selected_points[0]['x']], 
                                    y=[st.session_state.selected_points[0]['y']], 
                                    mode='markers',
                                    marker_symbol='star',
                                    marker_size=15,
                                    marker_color='black',
                                    name='selected'))
            
            # Select other Plotly events by specifying kwargs
            selected_points_scatter = plotly_events(fig, click_event=True, hover_event=False, select_event=False,
                                                    override_height=800, override_width=800)
                    
        
    with col2: # Polar distribution
        if len(df_unit_filtered):   
            col21, col22, col23 = st.columns((1,1,1))
            with col21:
                n_bins = st.slider("number of polar bins", 4, 32, 16, 4)
            with col22:
                polar_method = st.selectbox(label="polar method", options=['proportion in all neurons', 
                                                                           'proportion in significant neurons',
                                                                           'significant units weighted by r'],
                                            index=0)
            with col23:
                if_errorbar = st.checkbox("binomial 95% CI", value=True, disabled='in all neurons' not in polar_method)
                        
            fig = plot_polar(df_unit_filtered, x_name, y_name, polar_method, n_bins, if_errorbar)

            # # Select other Plotly events by specifying kwargs
            # selected_points_scatter = plotly_events(fig, click_event=True, hover_event=False, select_event=False,
            #                                         override_height=800, override_width=800)
            st.plotly_chart(fig, use_container_width=True)
        pass
    
                
    # -- bar plot unit sig proportion
    st.markdown(f'### Proportion of significant neurons for each variable (abs(t) > {sign_level}), errorbar = binomial 95% CI')
    fig = plot_unit_sig_prop_bar(df_unit_filtered, sign_level)
    st.plotly_chart(fig, use_container_width=True)
    
    # -- bar plot unit pure classifier --
    st.markdown(f'### Proportion of "pure" neurons (p_model < 0.01; polar classification), errorbar = binomial 95% CI')
    fig = plot_unit_pure_class_bar()
    st.plotly_chart(fig, use_container_width=True)
    
    with st.columns((1, 2))[0]:
        fig = plot_unit_class_scatter(x_name, y_name)
        st.plotly_chart(fig, use_container_width=True)

    
 
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
    
if if_profile:    
    p.stop()