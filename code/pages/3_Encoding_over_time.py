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

from streamlit_profiler import Profiler
p = Profiler()
p.start()

primary_keys = ['subject_id', 'session', 'insertion_number', 'unit']

def plot_linear_fitting_over_time(ds, model, paras, align_tos,
                                  para_stat='t',):
    
    fig = make_subplots(rows=len(paras), cols=len(align_tos), 
                        subplot_titles=align_tos,
                        shared_yaxes=True,
                        shared_xaxes=True,
                        vertical_spacing=0.02,
                        x_title='Time (s)',
                        )
    
    progress_bar = st.columns([1, 15])[0].progress(0, text='0%')
    
    # Retrieve unit_keys from dataset
    df_unit_keys = ds[primary_keys].to_dataframe().reset_index()
    df_filtered_unit_with_aoi = st.session_state.df_unit_filtered[primary_keys + ['area_of_interest']]
    
    # Add aoi to df_unit_keys; right join to apply the filtering
    df_unit_keys_filtered_and_with_aoi = df_unit_keys.merge(df_filtered_unit_with_aoi, on=primary_keys, how='right') 
    
    for row, para in enumerate(paras):
        for col, align_to in enumerate(align_tos):
            
            var_name = f'linear_fit_para_stats_aligned_to_{align_to}'
            t_name = f'linear_fit_t_center_aligned_to_{align_to}'
            ts = ds[t_name].values
            
            data_all = ds[var_name].sel(model=model, para=para, para_stat='t')
            
            for area, color in st.session_state.aoi_color_mapping.items():
                
                # Get unit_ind for this area
                unit_ind_this_area = df_unit_keys_filtered_and_with_aoi[df_unit_keys_filtered_and_with_aoi.area_of_interest == area].index
                
                # Select data for this area
                data_this_area = np.abs(data_all.sel(unit_ind=unit_ind_this_area).values)
                n = len(data_this_area)
                
                fig.add_trace(go.Scattergl(x=ts, 
                                            y=np.median(data_this_area, axis=0),
                                            mode='lines',
                                            line=dict(color=color),
                                            name=area,
                                            legendgroup=area,
                                            showlegend=col==0 and row ==0,
                                            hovertemplate=
                                                '%s, n = %s<br>' % (area, n) +
                                                '%{y:%2.1f} units, t > %{x:.2f}<br><extra></extra>',
                                            visible=True,
                                            ),
                            row=row+1, col=col+1)

            # fig.add_vline(x=2.0, line_color='gray', line_dash='dash',
            #             row=row+1, col=col+1)
            
            finished = (row * len(align_tos) + col + 1) / (len(align_tos) * len(paras))
            progress_bar.progress(finished, text=f'{finished:.0%}')


    for row, para in enumerate(paras):
        fig['layout'][f'yaxis{1 + row * len(align_tos)}']['title'] = para

    # fig.update_traces(line_width=3)
    fig.update_layout(width=min(2000, 400 + 270 * len(align_tos)), 
                      height=200 + 240 * len(paras),
                     font_size=17, hovermode='closest',
                     title= f'(Total number of units = {len(df_unit_keys_filtered_and_with_aoi)})',
                     title_x=0.01,
                     )
    fig.update_annotations(font_size=20)
        
    return fig
    
    pass
    

if __name__ == '__main__':

    if 'df' not in st.session_state: 
        init()
    
    
    # --- 1. Fetch zarr ---
    # try:   # Try local first (if in CO)
    #     zarr_store = '/root/capsule/data/datajoint_psth_linear_fit_over_timeall_linear_fit_over_time.zarr'
    #     ds_linear_fit_over_time = xr.open_zarr(zarr_store, consolidated=True)
    # except:  # Else, from s3
    s3_path = f's3://aind-behavior-data/Han/ephys/export/psth/all_linear_fit_over_time.zarr'
    fs = s3fs.S3FileSystem(anon=False)
    zarr_store = s3fs.S3Map(root=s3_path, s3=fs, check=True)
    ds_linear_fit_over_time = xr.open_zarr(zarr_store, consolidated=True)
    
    align_tos = ds_linear_fit_over_time.align_tos
    models = ds_linear_fit_over_time.linear_models
    
    # Prepare df
    df_period_linear_fit_all = st.session_state.df['df_period_linear_fit_all']
    unit_key_names = ['subject_id', 'session', 'insertion_number', 'unit']
    # Filter df and add area_of_interest to the index
    df_period_linear_fit_filtered = df_period_linear_fit_all.loc[st.session_state.df_unit_filtered.set_index(unit_key_names).index, :]
    df_aoi_filtered = st.session_state.df['df_ephys_units'].set_index(unit_key_names).loc[df_period_linear_fit_filtered.index, :]
    aoi_index = df_aoi_filtered.reset_index().set_index(unit_key_names + ['area_of_interest'])
    df_period_linear_fit_filtered.index = aoi_index.index

    plotly_font = lambda x: dict(xaxis_tickfont_size=x,
                                xaxis_title_font_size=x,
                                yaxis_tickfont_size=x,
                                yaxis_title_font_size=x,
                                legend_font_size=x,
                                legend_title_font_size=x,)
    
    with st.sidebar:    
        add_unit_filter()

        
    # --- t-distribution, compare areas ---
    st.markdown('#### :red[Linear fitting t-values over time]')

    st.markdown(
    """
    <style>
        .stMultiSelect [data-baseweb=select] span{
            max-width: 1000px;
        }
    </style>""",
    unsafe_allow_html=True,
    )

    cols = st.columns([1, 1, 1])
    
    _, selected_model = select_model(available_models=list(models.keys()), col=cols[0])

    available_paras_this_model = models[selected_model]
    
    
    selected_paras = cols[1].multiselect('Parameters',
                                         available_paras_this_model, 
                                         [p for p in available_paras_this_model if 'Q' in p 
                                          or p in ['reward', 'chosen_value', 'choice_this', 'choice_next', 'trial_normalized', 'firing_1_back']],
                                        )
    
    selected_align_to = cols[2].multiselect('Align to', 
                                                align_tos,
                                                ['go_cue', 'choice', 'iti_start'],
                                                key=f'align_to_linear_fit_over_time')
        
    if selected_model and selected_paras and selected_align_to:
        
        fig = plot_linear_fitting_over_time(ds=ds_linear_fit_over_time, 
                                            model=selected_model,
                                            paras=selected_paras,
                                            align_tos=selected_align_to)

        plotly_events(fig, override_height=fig.layout.height*1.1, override_width=fig.layout.width, click_event=False)
        
    p.stop()