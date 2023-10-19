import streamlit as st
import numpy as np
import pandas as pd

import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import s3fs
import os
import xarray as xr

from streamlit_plotly_events import plotly_events
import extra_streamlit_components as stx

from util import *
from util.selectors import (add_unit_filter, select_period, select_model, select_para)
from Home import init, _to_theta_r, select_t_sign_level, t_to_p, p_to_t

if_debug = False
if if_debug:
    from streamlit_profiler import Profiler
    p = Profiler()
    p.start()

primary_keys = ['subject_id', 'session', 'insertion_number', 'unit']
plot_settings = {'go_cue': {'win': [-1, 3], 'others': {'iti_start (median)': 2}},  # Plot window and other time points
                'choice': {'win': [-1, 3], 'others': {'iti_start (median)': 2}},
                'iti_start': {'win': [-2.5, 6], 'others': {'iti_start (median)': -2,
                                                           'next_trial_start (median)': 4.75}
                             },
                }

area_aggr_func_mapping = {    
                            # func, (min, max, step, default)
                            'median': lambda x: np.nanmedian(x, axis=0), 
                            'median abs()': lambda x: np.nanmedian(np.abs(x), axis=0),      
                            'mean': lambda x: np.nanmean(x, axis=0),
                            'mean abs()': lambda x: np.nanmean(np.abs(x), axis=0),
                            r'% significant units': lambda x: sum(np.abs(x) > sign_level) / len(x) * 100,  # para_stat must be 't'
                            }


def hash_xarray(ds):
    return hash(1)  # Fixed hash because I only have one dataset here

@st.cache_data(ttl=3600, hash_funcs={xr.Dataset: hash_xarray})
def _get_data_from_zarr(ds, var_name, model, para_stat):
    return ds[var_name].sel(model=model, para_stat=para_stat).values  # Get values from zarr here to avoid too much overhead
    
def plot_linear_fitting_over_time(ds, model, paras, align_tos,
                                  para_stat='t', aggr_func='median',
                                  sync_y=True):
    
    fig = make_subplots(rows=len(paras), cols=len(align_tos), 
                        subplot_titles=align_tos,
                        shared_yaxes=True,
                        shared_xaxes=True,
                        vertical_spacing=0.03,
                        x_title='Time (s)', 
                        column_widths=[col['win'][1] - col['win'][0] 
                                       for col in plot_settings.values()]
                        )
    
    progress_bar = st.columns([1, 15])[0].progress(0, text='0%')
    
    # Retrieve unit_keys from dataset
    df_unit_keys = ds[primary_keys].to_dataframe().reset_index()
    df_filtered_unit_with_aoi = st.session_state.df_unit_filtered[primary_keys + ['area_of_interest']]
    
    # Add aoi to df_unit_keys; right join to apply the filtering
    df_unit_keys_filtered_and_with_aoi = df_unit_keys.merge(df_filtered_unit_with_aoi, on=primary_keys, how='right') 
       
    for col, align_to in enumerate(align_tos):
        var_name = f'linear_fit_para_stats_aligned_to_{align_to}'
        t_name = f'linear_fit_t_center_aligned_to_{align_to}'
        ts = ds[t_name].values       
         
        data_all = _get_data_from_zarr(ds, var_name, model, para_stat)
            
        for row, para in enumerate(paras):       
            for area, color in st.session_state.aoi_color_mapping.items():
                # Get unit_ind for this area
                unit_ind_this_area = df_unit_keys_filtered_and_with_aoi[df_unit_keys_filtered_and_with_aoi.area_of_interest == area].unit_ind
                
                if not len(unit_ind_this_area): 
                    continue
                
                # Select data for this area
                data_this = data_all[unit_ind_this_area, 
                                     :, 
                                     list(ds.para).index(para),
                                     ]
                n = len(data_this)
                
                # Apply aggregation function
                y = area_aggr_func_mapping[aggr_func](data_this)
                
                fig.add_trace(go.Scattergl(x=ts, 
                                            y=y,
                                            mode='lines',
                                            line=dict(color=color),
                                            name=area,
                                            legendgroup=area,
                                            showlegend=col==0 and row ==0,
                                            hovertemplate=
                                                '%s, n = %s<br>' % (area, n) +
                                                '%s: %%{y:.4g}<br>'% (para) +
                                                '%%{x} s @ %s<br>' % (align_to) +
                                                '<extra></extra>',
                                            visible=True,
                                            ),
                            row=row+1, col=col+1)

            # Add type I error
            if aggr_func == r'% significant units':
                fig.add_hline(y=t_to_p(sign_level) * 100, row=row+1, col=col+1, 
                              line_dash='dash', line_color='black', line_width=1)
            
            # Add indicators for other time points
            for other_name, other_time in plot_settings[align_to]['others'].items():
                fig.add_vline(x=other_time, line_color='gray', line_dash='dash',
                              row=row+1, col=col+1)
            fig.add_vline(x=0, line_color='black', line_dash='solid',
                            name=align_to,
                            row=row+1, col=col+1)
                            
            finished = (col * len(paras) + row + 1) / (len(align_tos) * len(paras))
            progress_bar.progress(finished, text=f'{finished:.0%}')
            
        # set x range
        fig.update_xaxes(range=plot_settings[align_to]['win'], row=row+1, col=col+1)   
        
        if sync_y:
            y_min = 0
            y_max = 0
            for data in fig.data:
                y_min = min(y_min, np.nanmin(data.y))
                y_max = max(y_max, np.nanmax(data.y))
            
            for row in range(len(paras)):
                fig.update_yaxes(range=[y_min, y_max * 1.1], row=row+1, col=col+1)


    for row, para in enumerate(paras):
        fig['layout'][f'yaxis{1 + row * len(align_tos)}']['title'] = para

    # fig.update_traces(line_width=3)
    fig.update_layout(width=min(2000, 400 + 300 * len(align_tos)), 
                      height=200 + 240 * len(paras),
                     font_size=17, hovermode='closest',
                     title= f'{aggr_func} {f"(p < {t_to_p(sign_level):.2g})" if aggr_func == r"% significant units" else "of {para_stat}"}' 
                            f' (N = {len(df_unit_keys_filtered_and_with_aoi)})' +
                            f'',
                     title_x=0.01,
                     )
    fig.update_annotations(font_size=20)
        
    return fig
    

def plot_beta_auto_corr(ds, model, align_tos, paras,
                        down_sample_t=2,
                        sync_y=False):
    
    
    col_ratio = [col['win'][1] - col['win'][0] + 2
                     for col in plot_settings.values()]
        
    # Retrieve unit_keys from dataset
    df_unit_keys = ds[primary_keys].to_dataframe().reset_index()
    df_filtered_unit_with_aoi = st.session_state.df_unit_filtered[primary_keys + ['area_of_interest']]
    
    # Add aoi to df_unit_keys; right join to apply the filtering
    df_unit_keys_filtered_and_with_aoi = df_unit_keys.merge(df_filtered_unit_with_aoi, on=primary_keys, how='right') 
    unit_ind_filtered = df_unit_keys_filtered_and_with_aoi.unit_ind
    
    st.markdown(f'##### (N = {len(df_unit_keys_filtered_and_with_aoi)})')
    
    figs = [[None for _ in range(len(paras))] for _ in range(len(align_tos))]
    
    for col, align_to in enumerate(align_tos):
        var_name = f'linear_fit_para_stats_aligned_to_{align_to}'
        t_name = f'linear_fit_t_center_aligned_to_{align_to}'
        ts = ds[t_name].values[::down_sample_t]
         
        betas = _get_data_from_zarr(ds, var_name, model, 'beta')
  
        for row, para in enumerate(paras):
            # compute correlation matrix
            betas_down_sample = betas[unit_ind_filtered, 
                                      ::down_sample_t, 
                                      list(ds.para).index(para)]
            corr_matrix = np.corrcoef(betas_down_sample.T)
            np.fill_diagonal(corr_matrix, np.nan)  # Remove diagonal
            max_val = np.nanmax(np.abs(corr_matrix))
            
            # generate figure
            fig = go.Figure()            
            fig.add_trace(go.Heatmap(
                                    z=corr_matrix,
                                    zmin=-max_val, 
                                    zmax=max_val,
                                    x=ts,
                                    y=ts,
                                    colorscale='RdBu_r',
                                    ),
                         )
              
            # Add indicators for other time points
            for other_name, other_time in plot_settings[align_to]['others'].items():
                fig.add_shape(
                        type="line",
                        x0=other_time, x1=other_time, 
                        y0=plot_settings[align_to]['win'][0], y1=plot_settings[align_to]['win'][1],
                        line=dict(color="black", dash='dash'),
                    )
                fig.add_shape(
                        type="line",
                        x0=plot_settings[align_to]['win'][0], x1=plot_settings[align_to]['win'][1],
                        y0=other_time, y1=other_time, 
                        line=dict(color="black", dash='dash'),
                    )
            fig.add_shape(
                    type="line",
                    x0=0, x1=0, 
                    y0=plot_settings[align_to]['win'][0], y1=plot_settings[align_to]['win'][1],
                    line=dict(color="black", dash='solid'),
                )
            fig.add_shape(
                    type="line",
                    x0=plot_settings[align_to]['win'][0], x1=plot_settings[align_to]['win'][1],
                    y0=0, y1=0, 
                    line=dict(color="black", dash='solid'),
                )               
                                       
            # set plot range
            fig.update_xaxes(range=plot_settings[align_to]['win'])  
            fig.update_yaxes(range=plot_settings[align_to]['win']) 

            # for row, para in enumerate(paras):
            #     fig['layout'][f'yaxis{1 + row * len(align_tos)}']['title'] = para

            # fig.update_traces(line_width=3)
            
            fig.update_layout(
                            xaxis_title=f'Time to {align_to} (s)',
                            title=f'{para}',
                            width=90*col_ratio[col]*0.8, 
                            height=88*col_ratio[col]*0.8,
                            font_size=20, 
                            hovermode='closest',
                            )
            
            figs[col][row] = fig
            
    # Iterate over row first to align subplots horizontally
    for row, para in enumerate(paras):
        cols = st.columns(col_ratio + [2])
        for col, align_to in enumerate(align_tos):
            with cols[col]:
                plotly_events(figs[col][row],
                            override_height=fig.layout.height*1.1, 
                            override_width=fig.layout.width, click_event=False)
        
    return



if __name__ == '__main__':

    if 'df' not in st.session_state: 
        init()
        
    with st.sidebar:    
        add_unit_filter()

    
    # --- 1. Fetch zarr ---
    try:   # Try local first (if in CO)
        assert 0
        zarr_store = '/root/capsule/data/datajoint_psth_linear_fit_over_timeall_linear_fit_over_time.zarr'
        ds_linear_fit_over_time = xr.open_zarr(zarr_store, consolidated=True)
    except:  # Else, from s3
        s3_path = f's3://aind-behavior-data/Han/ephys/export/psth/all_linear_fit_over_time.zarr'
        fs = s3fs.S3FileSystem(anon=False)
        zarr_store = s3fs.S3Map(root=s3_path, s3=fs, check=True)
        ds_linear_fit_over_time = xr.open_zarr(zarr_store, consolidated=True)
    
    align_tos = ds_linear_fit_over_time.align_tos
    models = ds_linear_fit_over_time.linear_models
    
    plotly_font = lambda x: dict(xaxis_tickfont_size=x,
                                xaxis_title_font_size=x,
                                yaxis_tickfont_size=x,
                                yaxis_title_font_size=x,
                                legend_font_size=x,
                                legend_title_font_size=x,)
    
    st.markdown('### Select a tab here 👇')
    chosen_id = stx.tab_bar(data=[
                                stx.TabBarItemData(id="tab1", title="1. Fitting stats", description=""),
                                stx.TabBarItemData(id="tab2", title="2. Coding directions", description=""),
                                ], 
                            default="tab2")
    
    if chosen_id == 'tab1':
        st.markdown('#### :red[Linear fitting over time]')

        st.markdown(
        """
        <style>
            .stMultiSelect [data-baseweb=select] span{
                max-width: 1000px;
            }
        </style>""",
        unsafe_allow_html=True,
        )

        # --- 2. Settings ---
        cols = st.columns([1, 1, 1])
        
        selected_model = cols[0].selectbox('Linear model',
                                            list(models.keys()), 
                                            1,
                                        )
        selected_align_to = cols[0].multiselect('Align to', 
                                                    align_tos,
                                                    ['go_cue', 'choice', 'iti_start'],
                                                    key=f'align_to_linear_fit_over_time')

        available_paras_this_model = models[selected_model]
        
        selected_paras = cols[1].multiselect('Parameters',
                                            available_paras_this_model, 
                                            [p for p in available_paras_this_model if 'Q' in p 
                                            or p in ['reward', 'chosen_value', 'choice_this', 'choice_next', 'trial_normalized', 'firing_1_back']],
                                            )
        
        cc = cols[1].columns([1, 1])
        selected_agg_func = cc[0].selectbox('Aggr func',
                                            [r'% significant units', 'median abs()', 'mean abs()', 'median', 'mean'], 
                                            0
                                            )
        if selected_agg_func != r'% significant units':
            selected_para_stat = cc[1].selectbox('Statistic',
                                                ds_linear_fit_over_time.para_stat.values, 
                                                list(ds_linear_fit_over_time.para_stat).index('t'),
                                                )
        else:
            selected_para_stat = 't'
            sign_level = select_t_sign_level(col=cc[1])
            
        
        # --- 3. Plot ---
        if selected_model and selected_paras and selected_align_to:
            
            fig = plot_linear_fitting_over_time(ds=ds_linear_fit_over_time, 
                                                model=selected_model,
                                                paras=selected_paras,
                                                para_stat=selected_para_stat,
                                                aggr_func=selected_agg_func,
                                                align_tos=selected_align_to,
                                                sync_y=cols[1].checkbox('Sync y-axis', True)
            )

            plotly_events(fig, override_height=fig.layout.height*1.1, 
                        override_width=fig.layout.width, click_event=False)
            
    elif chosen_id == 'tab2':        
        cols = st.columns([1, 1, 1])
        selected_model = cols[0].selectbox('Linear model',
                                            list(models.keys()), 
                                            1,
                                        )
        selected_align_to = cols[0].multiselect('Align to', 
                                                    align_tos,
                                                    ['go_cue', 'choice', 'iti_start'],
                                                    key=f'align_to_linear_fit_over_time')

        available_paras_this_model = models[selected_model]
        
        selected_paras = cols[1].multiselect('Parameters',
                                            available_paras_this_model, 
                                            [p for p in available_paras_this_model if 'Q' in p 
                                            or p in ['reward', 'chosen_value', 'choice_this', 'choice_next', 'trial_normalized', 'firing_1_back']],
                                            )
        
        align_to = selected_align_to[0]        

        plot_beta_auto_corr(ds=ds_linear_fit_over_time, 
                                  model=selected_model, 
                                  paras=selected_paras, 
                                  align_tos=selected_align_to)
        
    if if_debug:
        p.stop()