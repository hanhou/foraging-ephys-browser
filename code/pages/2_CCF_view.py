import plotly.graph_objects as go
import numpy as np
from PIL import Image, ImageColor
import pandas as pd
import scipy
from plotly.utils import image_array_to_data_uri
import plotly.express as px

import streamlit as st
ss = st.session_state
from streamlit_plotly_events import plotly_events
from util.streamlit_util import add_unit_filter, unit_plot_settings, add_unit_selector, draw_selected_units

from Home import init, select_t_sign_level, pure_unit_color_mapping

import importlib
uplf = importlib.import_module('.1_Linear_model_comparison', package='pages')

import nrrd

    
CCF_RESOLUTION = 25

@st.cache_data(ttl=24*3600)
def _get_min_max():
    x_gamma_all = np.abs(ss.df_unit_filtered_merged[column_to_map] ** size_gamma)
    return np.percentile(x_gamma_all, 5), np.percentile(x_gamma_all, 95)

def _size_mapping(x):
    x = x / np.quantile(x, 0.95)
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



# @st.cache_data(ttl=24*3600, experimental_allow_widgets=True, show_spinner=True)
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
    fig.add_trace(go.Scattergl(
        x=[300, 300, 300],
        y=[300, 600, 900],
        mode='text',
        text=message,
        textfont=dict(size=20),
        textposition='bottom right',
        showlegend=False
    ))
    
    # -- overlay edges
    xx, yy = edges
    fig.add_trace(go.Scattergl(x=yy * CCF_RESOLUTION, y=xx * CCF_RESOLUTION, 
                             mode='markers',
                             marker={'color': 'rgba(0, 0, 0, 0.3)', 'size': 2},
                             hoverinfo='skip',
                             showlegend=False,
                             ))
    # fig.update_layout(clickmode='none', selectmode='none')
    return fig



@st.cache_data(persist='disk', show_spinner=False)
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


@st.cache_data(ttl=24*3600)
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
   

def draw_ccf_heatmap(fig, units_to_overlay, slice_name):
    
    heatmap = scipy.stats.binned_statistic_2d(x=units_to_overlay.x, y=units_to_overlay.y,
                                            values=units_to_overlay.z, 
                                            statistic=heatmap_aggr_func, 
                                            bins=[np.arange(units_to_overlay.x.min() - heatmap_bin_size, units_to_overlay.x.max() + heatmap_bin_size, heatmap_bin_size),
                                                  np.arange(units_to_overlay.y.min() - heatmap_bin_size, units_to_overlay.y.max() + heatmap_bin_size, heatmap_bin_size)])
    
    heatmap_smoothed = _smooth_heatmap(heatmap.statistic, sigma=heatmap_smooth)
    
    # Build region name
    heatmap_region = np.full(heatmap_smoothed.shape, '', dtype=object)
    ind_xs, ind_ys = np.where(~np.isnan(heatmap_smoothed))
    for ind_x, ind_y in zip(ind_xs, ind_ys):
        ind_x_ccf = int(np.round(heatmap.x_edge[ind_x] / CCF_RESOLUTION))
        ind_y_ccf = int(np.round(heatmap.y_edge[ind_y] / CCF_RESOLUTION))
        heatmap_region[ind_x, ind_y] = (slice_name[ind_y_ccf, ind_x_ccf])
        
    zmin, zmax = (-heatmap_color_range, heatmap_color_range) if if_bi_directional_heatmap else (heatmap_color_range[0], heatmap_color_range[1])
        
    fig.add_trace(go.Heatmap(z=heatmap_smoothed.T, 
                            x=heatmap.x_edge, 
                            y=heatmap.y_edge,
                            zmin=zmin,
                            zmax=zmax,
                            hoverongaps=False,
                            hovertemplate = "%{customdata[0]}" + 
                                            "<br>%s (%s): %%{z}" % (heatmap_aggr_name, column_to_map_name) + 
                                            "<extra></extra>",
                            customdata=heatmap_region.T,
                            colorscale='RdBu' if if_bi_directional_heatmap else 'plasma', 
                            colorbar=dict(orientation="h", len=0.3)
                            ))
    
    return

def draw_ccf_units(fig, units_to_overlay, selected):
    units_to_overlay['size'] = _size_mapping(np.abs(units_to_overlay.z))
    
    if np.sum(units_to_overlay.z < 0) == 0:  # All positive values
        units_to_overlay['color'] = 'rgba(0, 0, 0, 0.8)'
    else:
        # red (for unit_stats: ipsi), positive: blue (for unit_stats: contra)
        units_to_overlay['color'] = 'rgba(255, 0, 0, 0.8)'
        units_to_overlay.loc[units_to_overlay.z >=0, 'color'] = 'rgba(0, 0, 255, 0.8)'
        
    # If there are selected dots, put unselcted dots to lighter color
    if len(selected):
        units_to_overlay.loc[~units_to_overlay.reset_index().set_index(ss.unit_key_names).index.
                             isin(selected.reset_index().set_index(ss.unit_key_names).index), 'color'] = 'rgba(255, 255, 255, 0.2)'
    
    units_to_overlay = units_to_overlay.reset_index()
    fig.add_trace(go.Scattergl(x=units_to_overlay.x, 
                               y=units_to_overlay.y,
                               mode = 'markers',
                                marker_size = _size_mapping(np.abs(units_to_overlay.z)),
                                # continuous_color_scale = 'RdBu',
                                marker = {'color': units_to_overlay.color, 
                                          'line': dict(width=1.2, color='white')},
                                hovertemplate= '"%{customdata[0]}"' + 
                                                '<br>%{text}' +
                                                '<br>%s = %%{customdata[1]}' % column_to_map_name +
                                                '<br>uid = %{customdata[2]}' +
                                                '<extra></extra>',
                                text=units_to_overlay.annotation,
                                customdata=np.stack((units_to_overlay.area_of_interest, 
                                                     units_to_overlay.z, 
                                                     units_to_overlay.uid), axis=-1),
                                showlegend=False,
                                unselected=dict(marker_color='rgba(255, 255, 255, 0.2)'),
                                ))
        
    return


def plot_coronal_slice_unit(ccf_x, coronal_slice_thickness, *args):
    fig = go.Figure()

    # -- ccf annotation --
    coronal_slice, coronal_slice_name, coronal_edges = get_slice('coronal', ccf_x)
    
    if ss.if_flip:
        max_x = int(np.ceil(5700 / CCF_RESOLUTION))
        coronal_slice = coronal_slice[:, :max_x, :]
        coronal_slice_name = coronal_slice_name[:, :max_x, :]
        coronal_edges = [coord[coronal_edges[1] < max_x] for coord in coronal_edges]
        
    message = [ f'AP ~ {ccf_x_to_AP(ccf_x)} mm', 
                f'Slice thickness = {coronal_slice_thickness} um']
    draw_ccf_annotations(fig, coronal_slice, coronal_slice_name, 
                                coronal_edges, message)
    
    # -- overlayed units --
    units_to_overlay = ss.df_unit_filtered_merged.query(f'{ccf_x - coronal_slice_thickness/2} < ccf_x and ccf_x <= {ccf_x + coronal_slice_thickness/2}'
                )[['ccf_z', 'ccf_y', 'annotation', column_to_map]
                  ].rename(columns={'ccf_z': 'x', 'ccf_y': 'y', column_to_map: 'z'})
    
    if len(units_to_overlay):
        if if_take_abs:
            units_to_overlay.z = np.abs(units_to_overlay.z)

        if ss.if_flip:
            units_to_overlay.x[units_to_overlay.x > 5700] = 5700 * 2 - units_to_overlay.x[units_to_overlay.x > 5700]
            
        if ss.if_ccf_plot_heatmap:
            try:
                draw_ccf_heatmap(fig, units_to_overlay, coronal_slice_name)
            except:
                pass

        if ss.if_ccf_plot_scatter:
            draw_ccf_units(fig, units_to_overlay, selected=ss.df_selected_from_ccf_coronal)
        
    fig.update_layout(width=800 if ss.if_flip else 1100, 
                      height= 1000,
                      xaxis_range=[0, 5700 if ss.if_flip else 5700*2],
                      yaxis_range=[8000, -10],
                      xaxis_title='ccf_z (left -> right)',
                      yaxis_title='ccf_y (superior -> inferior)',
                      font=dict(size=20),
                      hovermode='closest',
                      dragmode='zoom',
                      title=f'{heatmap_aggr_name} of {column_to_map_name}' if heatmap_aggr_name is not None else column_to_map_name + 
                            (f' [{uplf.period_name_mapper[column_to_map[0]]}]' if isinstance(column_to_map, tuple) else ''),
                      title_font_size=20,
                      )
    
    # st.plotly_chart(fig, use_container_width=True)
    # st.pyplot(fig)
    return fig


def plot_saggital_slice_unit(ccf_z, saggital_slice_thickness, *args):
    fig = go.Figure()

    # -- ccf annotation --
    saggital_slice, saggital_slice_name, saggital_edges = get_slice('saggital', ccf_z)

    message = [f'ML ~ {ccf_z_to_ML(ccf_z)} mm', 
               f'Slice thickness = {saggital_slice_thickness} um']

    fig = draw_ccf_annotations(fig, saggital_slice, saggital_slice_name,
                               saggital_edges, message)

    # -- overlayed units --
    if len(ss.df_unit_filtered_merged):
        ss.df_unit_filtered_merged['ccf_z_in_slice_plot'] = ss.df_unit_filtered_merged['ccf_z']
        if ss.if_flip:
            to_flip_idx = ss.df_unit_filtered_merged['ccf_z_in_slice_plot'] > 5700
            to_flip_value = ss.df_unit_filtered_merged['ccf_z_in_slice_plot'][to_flip_idx]
            ss.df_unit_filtered_merged.loc[to_flip_idx, 'ccf_z_in_slice_plot'] = 2 * 5700 - to_flip_value
        
        # -- overlayed units --
        units_to_overlay = ss.df_unit_filtered_merged.query(
                    f'{ccf_z - saggital_slice_thickness/2} < ccf_z_in_slice_plot and ccf_z_in_slice_plot <= {ccf_z + saggital_slice_thickness/2}'
                    )[['ccf_x', 'ccf_y', 'annotation', column_to_map]
                    ].rename(columns={'ccf_x': 'x', 'ccf_y': 'y', column_to_map: 'z'})

    if len(units_to_overlay):
        if if_take_abs:
            units_to_overlay.z = np.abs(units_to_overlay.z)

        if ss.if_ccf_plot_heatmap:
            draw_ccf_heatmap(fig, units_to_overlay, saggital_slice_name)
        
        if ss.if_ccf_plot_scatter:             
            draw_ccf_units(fig, units_to_overlay, selected=ss.df_selected_from_ccf_saggital)
    
    fig.update_layout(width=1300 if ss.if_flip else 1000, 
                      height=1000,
                      xaxis_range=[0, 10000],
                      yaxis_range=[8000, -10],
                      xaxis_title='ccf_x (anterior -> posterior)',
                      yaxis_title='ccf_y (superior -> inferior)',
                      font=dict(size=20),
                      hovermode='closest',
                      title=f'{heatmap_aggr_name} of {column_to_map_name}' if heatmap_aggr_name is not None else column_to_map_name + 
                            (f'[{uplf.period_name_mapper[column_to_map[0]]}]' if isinstance(column_to_map, tuple) else ''),
                      title_font_size=20,
                      dragmode='zoom',
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


def _ccf_heatmap_available_aggr_funcs(column_to_map):
    
    if if_map_pure:
        return [r'% pure units']
    
    if isinstance(column_to_map, tuple) and column_to_map[2] == 't':  # Unit tuning
        return [r'% significant units', 'median', 'median (significant only)', 'mean', 'mean (significant only)']
    
    return ['median', 'mean']
        

def _ccf_heatmap_get_aggr_func_and_range(heatmap_aggr_name, column_to_map, value_to_map):
    
    if isinstance(column_to_map, tuple) and column_to_map[2] == 't':
        heatmap_range = (0.0, 10.0, 0.01, 5.0 if if_bi_directional_heatmap else (1.96, 5.0))
    elif if_map_pure:
        heatmap_range = (0, 100, 1, (5, 50))
    else:
        range_min = float(np.quantile(np.abs(value_to_map), 0.05))
        range_default = float(np.quantile(np.abs(value_to_map), 0.8))
        range_max = float(np.quantile(np.abs(value_to_map), 0.99))
        heatmap_range = (0.0, range_max, range_max/100, range_default if if_bi_directional_heatmap else (range_min, range_default))      

    heatmap_aggr_func_mapping = {    
                                 # func, (min, max, step, default)
                                'median': ('median', heatmap_range),        
                                'median (significant only)': (lambda x: np.median(x[np.abs(x) >= sign_level]), heatmap_range),      
                                'mean': ('mean', heatmap_range),
                                'mean (significant only)': (lambda x: np.mean(x[np.abs(x) >= sign_level]), heatmap_range),
                                
                                r'% significant units': (lambda x: sum(np.abs(x) >= sign_level) / len(x) * 100 if len(x) else np.nan, 
                                                        (5, 100, 5, (5, 80))),
                                
                                'number of units': (lambda x: len(x) if len(x) else np.nan, 
                                                    (0, 50, 5, (0, 20))),
                                
                                r'% pure units': (lambda x: sum(x) / len(x) * 100, heatmap_range),
                                
                                # 'max': ('max', (0.0, 15.0, 1.0, (0.0, 5.0))),
                                # 'min': ('min', (0.0, 15.0, 1.0, (0.0, 5.0))),
                                }
                                
    return heatmap_aggr_func_mapping[heatmap_aggr_name][0], heatmap_aggr_func_mapping[heatmap_aggr_name][1]


def select_para_of_interest(prompt="Map what to CCF?", suffix='',
                            default_model='dQ, sumQ, rpe, C*2, R*5, t',
                            default_period='iti_all',
                            default_paras=None,):
    
    type_to_map = st.selectbox(prompt,
                            ['unit tuning', 'unit stats'],
                            key=f'type_to_map{suffix}',
                            index=0)
    
    if type_to_map == 'unit tuning':
        
        _, model = uplf.select_model(label='which model',
                                     suffix=suffix,
                                     default_model=default_model,
                                    )
        _, period = uplf.select_period(multi=False,
                                       suffix=suffix,
                                       default_period=default_period,
                                       label='which period')
        
        df_this_model = uplf.df_period_linear_fit_all.iloc[:, uplf.df_period_linear_fit_all.columns.get_level_values('multi_linear_model') == model]

        cols= st.columns([1, 1])
        stat = cols[0].selectbox("which statistic",
                                ['t', 'beta', 'model_r2', 'model_bic'] + list(pure_unit_color_mapping.keys()), 
                                0,
                                key=f'stat{suffix}')  # Could be model level stats, like 'model_r2'
        available_paras_this_model = [p for p in uplf.para_name_mapper if p in 
                                    df_this_model.columns[df_this_model.columns.get_level_values('stat_name') == stat
                                                        ].get_level_values('var_name').unique()]
        if available_paras_this_model:
            _, para = uplf.select_para(multi=False,
                                       suffix=suffix,
                                        available_paras=available_paras_this_model,
                                        default_paras=available_paras_this_model[0] if default_paras is None else default_paras,
                                        label='which variable',
                                        col=cols[1])
        else:
            para = ''
        
        column_to_map = (period, model, stat, para)
        column_to_map_name = f'{stat}_{uplf.para_name_mapper[para]}' if para != '' else stat
        if_map_pure = 'pure' in column_to_map[2]
        
    elif type_to_map == 'unit stats':
        column_to_map = st.selectbox("which stat", 
                                    ss.ccf_stat_names, 
                                    index=ss.ccf_stat_names.index('avg_firing_rate'),
                                    key=f'stat{suffix}')
        column_to_map_name = column_to_map
        if_map_pure = False
    
    column_selected = ss.df_unit_filtered_merged.loc[:, [column_to_map]]
    column_selected.dropna(inplace=True)
    
    if_take_abs = st.checkbox("Use abs()?", value=False, key='abs'+suffix) if np.any(column_selected < 0) else False

    if if_take_abs:
        column_selected = np.abs(column_selected)
        
    return dict(column_to_map=column_to_map, column_to_map_name=column_to_map_name, 
                if_map_pure=if_map_pure, if_take_abs=if_take_abs,
                column_selected=column_selected)
    
    
def _update_selected_index_from_ccf(dict_selected, view):
    if not len(dict_selected): return
    cols = ['ccf_x', 'ccf_y'] if view == 'saggital' else ['ccf_z', 'ccf_y']

    df_selected = pd.DataFrame(dict_selected).query('curveNumber > 2')[['x', 'y']]
    df_selected.columns = cols
    df_selected = ss.df_unit_filtered_merged[['ccf_x', 'ccf_y', 'ccf_z']].reset_index(
        ).merge(df_selected, how='inner').set_index(ss.unit_key_names)
    
    # If selected units change, rerun the whole app
    if not (set(df_selected.index) == set(ss[f'df_selected_from_ccf_{view}'].index)):
        ss[f'df_selected_from_ccf_{view}'] = df_selected
        st.experimental_rerun()
        

if __name__ == '__main__':
    
    if 'df' not in ss: 
        init()

    with st.sidebar:    
        try:
            add_unit_filter()
        except:
            st.experimental_rerun()

        with st.expander("CCF view settings", expanded=True):
            
            with st.expander('Value mapping', expanded=True):
                para = select_para_of_interest()
                column_to_map, column_to_map_name, if_map_pure, if_take_abs, column_selected = \
                    para['column_to_map'], para['column_to_map_name'], para['if_map_pure'], para['if_take_abs'], para['column_selected']

                    
                # Add a small histogram
                if column_to_map != 'number_units' and not if_map_pure:
                    counts, bins = np.histogram(column_selected, bins=100)        
                    fig = px.bar(x=bins[1:], y=counts)
                    # fig.add_vrect(x0=user_num_input[0], x1=user_num_input[1], fillcolor='red', opacity=0.1, line_width=0)
                    fig.update_layout(showlegend=False, height=100, 
                                    yaxis=dict(visible=False),
                                    xaxis=dict(title=''),
                                    margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                    
            with st.expander('Unit scatter', expanded=True):
                if column_to_map != 'number_units' and not if_map_pure:
                    ss.if_ccf_plot_scatter = st.checkbox("Draw units", 
                                                      value=ss.if_ccf_plot_scatter if 
                                                            'if_ccf_plot_scatter' in ss else True)
                else:
                    ss.if_ccf_plot_scatter = False
                
                if ss.if_ccf_plot_scatter:       
                    with st.expander("Unit settings", expanded=True):
                        size_range = st.slider("size range", 0, 100, (0, 50))
                        size_gamma = st.slider("gamma", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
            
            with st.expander('Heatmap', expanded=True):
                ss.if_ccf_plot_heatmap = st.checkbox("Draw heatmap", 
                                                  value=ss.if_ccf_plot_heatmap if 
                                                        'if_ccf_plot_heatmap' in ss else True)
                if not ss.if_ccf_plot_heatmap:
                    heatmap_aggr_name = None
                else:
                    with st.expander("Heatmap settings", expanded=True):
                        if column_to_map != 'number_units':
                            available_aggr_funcs  = _ccf_heatmap_available_aggr_funcs(column_to_map)
                            heatmap_aggr_name = st.selectbox("aggregate function", 
                                                            available_aggr_funcs, 
                                                            index=0)
                        else:
                            heatmap_aggr_name = 'number of units'
                        
                        if 'sign' in heatmap_aggr_name or if_map_pure: 
                            sign_level = select_t_sign_level(st)

                        if_bi_directional_heatmap = (any(column_selected < 0) + any(column_selected > 0)) == 2 and r'%' not in heatmap_aggr_name 
                        heatmap_aggr_func, heatmap_color_ranges = _ccf_heatmap_get_aggr_func_and_range(heatmap_aggr_name, column_to_map, column_selected)
                        heatmap_color_range = st.slider(f"Heatmap color range", 
                                                        heatmap_color_ranges[0], heatmap_color_ranges[1],
                                                        step=heatmap_color_ranges[2], value=heatmap_color_ranges[3])
                        
                        heatmap_bin_size = st.slider("Heatmap bin size", 25, 500, step=25, value=100)
                        heatmap_smooth = st.slider("Heatmap smooth factor", 0.0, 2.0, step=0.1, value=1.0)
                
            ss.if_flip = st.checkbox("Flip to left hemisphere", value=True)  
            
        add_unit_selector()      


    # --- coronal slice ---
    col_coronal, col_saggital = st.columns((1, 1.8)) if ss.if_flip else st.columns((1, 1))
    with col_coronal:
        col = st.columns([0.18 if ss.if_flip else 0.085, 1, 0.04 if ss.if_flip else 0.05])[1]
        ccf_z = col.slider("Saggital slice at (ccf_z)", min_value=0, max_value=5700 if ss.if_flip else 10800, 
                                        value=ss.ccf_z if 'ccf_z' in ss else 5100, 
                                        step=100)       # whole ccf @ 25um [528x320x456] 
        saggital_thickness = col.slider("Slice thickness (LR)", min_value= 100, max_value=5000, step=50, value=700)
        
        container_coronal = st.container()
        
    # --- saggital slice ---
    with col_saggital:
        col = st.columns([0.13 if ss.if_flip else 0.09, 1, 0.15 if ss.if_flip else 0.11])[1]
        ccf_x = col.slider("Coronal slice at (ccf_x)", min_value=0, max_value=10000, value=3500, step=100) # whole ccf @ 25um [528x320x456] 
        # st.markdown(f'##### AP relative to Bregma ~ {ccf_x_to_AP(ccf_x): .2f} mm') 
        coronal_thickness = col.slider("Slice thickness (AP)", min_value= 100, max_value=5000, step=50, value=700)
        
        container_saggital = st.container()

    container_unit_all_in_one = st.container()


    with container_coronal:
        fig = plot_coronal_slice_unit(ccf_x, coronal_thickness,) #, [size_to_map, size_gamma, size_range], aggrid_outputs, ccf_z, saggital_thickness)
        fig.add_vline(x=ccf_z, line_width=1)
        fig.add_vline(x=max(ccf_z - saggital_thickness/2, fig.layout.xaxis.range[0]), line_width=1, line_dash='dash')
        fig.add_vline(x=min(ccf_z + saggital_thickness/2, fig.layout.xaxis.range[1]), line_width=1, line_dash='dash')

        selected_points_from_coronal = plotly_events(fig, click_event=True, hover_event=False, select_event=True,
                                              override_height=fig.layout.height*1.1)
        _update_selected_index_from_ccf(selected_points_from_coronal, 'coronal')


    with container_saggital:
        fig = plot_saggital_slice_unit(ccf_z, saggital_thickness) #, [size_to_map, size_gamma, size_range], aggrid_outputs, ccf_x, coronal_thickness)
        selected_points_from_saggital = plotly_events(fig, click_event=True, hover_event=False, select_event=True,
                                              override_height=fig.layout.height*1.1)

        _update_selected_index_from_ccf(selected_points_from_saggital, 'saggital')
    

    # --- Draw selected units ---
    st.markdown("---")
    with st.columns([1, 2])[0]:
        if_draw_units = unit_plot_settings(default_source='ccf_coronal', need_click=True)
    
    if if_draw_units:
        draw_selected_units()
