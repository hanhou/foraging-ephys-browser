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
from util.plotly_util import add_plotly_errorbar

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

# Define default time epoch over which we average beta to get coding direction
# Note that the align_to here could be independent from the align_to in the psth
coding_direction_beta_aver_epoch = {
    'dQ': dict(align_to='iti_start', win=[0, 4]),
    'sumQ': dict(align_to='iti_start', win=[0, 4]),
    #'reward': dict(align_to='choice', win=[0, 2]),
    'reward': dict(align_to='iti_start', win=[0, 4]),
    'chosen_value': dict(align_to='choice', win=[0, 2]),
    'choice_this': dict(align_to='choice', win=[-0.2, 0.5]),
    }

user_color_mapping = px.colors.qualitative.Plotly  # If not ccf color, use this color mapping


def hash_xarray(ds):
    return hash(1)  # Fixed hash because I only have one dataset here

@st.cache_data(ttl=3600*24, hash_funcs={xr.Dataset: hash_xarray})
def _get_data_from_zarr(ds, var_name, model, para_stat):
    return ds[var_name].sel(model=model, para_stat=para_stat).values  # Get values from zarr here to avoid too much overhead
    
def plot_linear_fitting_over_time(ds, model, paras, align_tos,
                                  para_stat='t', aggr_func='median',
                                  if_95_CI=True,
                                  if_sync_y=True,
                                  if_use_ccf_color=True):
    
    fig = make_subplots(rows=len(paras), cols=len(align_tos), 
                        subplot_titles=align_tos,
                        shared_yaxes=True,
                        shared_xaxes=True,
                        vertical_spacing=0.03,
                        x_title='Time (s)', 
                        column_widths=[plot_settings[align_to]['win'][1] - plot_settings[align_to]['win'][0] 
                                       for align_to in align_tos]
                        )
    
    
    # Retrieve unit_keys from dataset
    df_unit_keys = ds[primary_keys].to_dataframe().reset_index()
    df_filtered_unit_with_aoi = st.session_state.df_unit_filtered[primary_keys + ['area_of_interest']]
    
    # Add aoi to df_unit_keys; right join to apply the filtering
    df_unit_keys_filtered_and_with_aoi = df_unit_keys.merge(df_filtered_unit_with_aoi, on=primary_keys, how='right') 
           
    progress_bar = st.columns([1, 15])[0].progress(0, text='0%')
    st.markdown(f'##### {aggr_func} {f"(p < {t_to_p(sign_level):.2g})" if aggr_func == r"% significant units" else f"of {para_stat}"}' 
                f' (N = {len(df_unit_keys_filtered_and_with_aoi)}, sliding window width = {win_width:.2g} s, step = {win_step:.2g} s)' +
                f'')
    
    for col, align_to in enumerate(align_tos):
        var_name = f'linear_fit_para_stats_aligned_to_{align_to}'
        t_name = f'linear_fit_t_center_aligned_to_{align_to}'
        ts = ds[t_name].values       
         
        data_all = _get_data_from_zarr(ds, var_name, model, para_stat)
            
        for row, para in enumerate(paras):       
            for i, (area, color) in enumerate(st.session_state.aoi_color_mapping.items()):
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
                
                color_this = color if if_use_ccf_color else user_color_mapping[i%len(user_color_mapping)]
                
                if not if_95_CI:
                    fig.add_trace(go.Scattergl(x=ts, 
                                                y=y,
                                                mode='lines',
                                                line=dict(color=color_this),
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
                else:   
                    binomial_95_CI = np.sqrt(y/100*(1-y/100)/n)*100 * 1.96  # 95% confidence interval
                    add_plotly_errorbar(x=pd.Series(ts), 
                                        y=pd.Series(y),
                                        err=pd.Series(binomial_95_CI), 
                                        color=color_this,
                                        name=area,
                                        mode='lines',
                                        showlegend=col==0 and row ==0,
                                        hovertemplate=
                                            '%s, n = %s<br>' % (area, n) +
                                            '%s: %%{y:.4g}<br>'% (para) +
                                            '%%{x} s @ %s<br>' % (align_to) +
                                            '<extra></extra>',
                                        visible=True,
                                        fig=fig, alpha=0.4, 
                                        subplot_specs=dict(col=col+1, row=row+1)
                                        )

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
        
        if if_sync_y:
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
    fig.update_layout(width=min(2000, 600 + 300 * len(align_tos)), 
                      height=200 + 400 * len(paras),
                     font_size=17, hovermode='closest',
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
    
    st.markdown(f'##### (N = {len(df_unit_keys_filtered_and_with_aoi)}, '
                f'sliding window width = {win_width:.2g} s, step = {win_step:.2g} s)')
    
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
                                    zauto=False,
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

@st.cache_data(ttl=60*60*24)
def _get_psth(psth_name, align_to, psth_grouped_by, select_units, lr_or_ic, right_unit_to_flip):
    t_range = {f't_to_{align_to}': slice(*plot_settings[align_to]['win'])}
    
    psth_mean, psth_sem = ds_psth[psth_name].sel(stat=['mean', 'sem'], 
                                                unit_ind=select_units,
                                                **t_range).values   
    
    ts = ds_psth[f't_to_{align_to}'].sel(**t_range).values
    group_name = ds_psth[f'psth_groups_{psth_grouped_by}'].values
    
    plot_spec = ds_psth[f'psth_setting_plot_spec_{psth_grouped_by}'].values
    if 'reward' in psth_grouped_by:  # Bug fix
        plot_spec = plot_spec[[1, 0, 3, 2]]
        
    # Flip lr to ic if needed
    if lr_or_ic == 'Ipsi and Contra':
        if 'choice' in psth_grouped_by:
            psth_mean[right_unit_to_flip] = psth_mean[right_unit_to_flip][:, [2, 3, 0, 1]]
            psth_sem[right_unit_to_flip] = psth_sem[right_unit_to_flip][:, [2, 3, 0, 1]]
            group_name = ['ipsi, no_rew', 'ipsi, rew', 'contra, no_rew', 'contra, rew']
        elif 'dQ' in psth_grouped_by:
            psth_mean[right_unit_to_flip] = np.flip(psth_mean[right_unit_to_flip], axis=1)
            psth_sem[right_unit_to_flip] = np.flip(psth_sem[right_unit_to_flip], axis=1)           
        
    
    return [psth_mean, psth_sem, ts, group_name, plot_spec]
    
@st.cache_data(ttl=60*60*24)
def _get_coding_direction(model, para, align_to, beta_aver_epoch, select_units):
    var_name = f'linear_fit_para_stats_aligned_to_{align_to}'
    t_name = f'linear_fit_t_center_aligned_to_{align_to}'
    
    aver_betas = ds_linear_fit_over_time[var_name].sel(model=model,
                                                       para=para, 
                                                       para_stat='beta',
                                                       unit_ind=select_units,
                                                       **{t_name: slice(*beta_aver_epoch)}
                                                       ).mean(dim=t_name).values
    
    return aver_betas / np.sqrt(np.sum(aver_betas**2))


def compute_psth_proj_on_CD(psth, psth_sem, coding_direction, if_error_bar):
    # Handle PSTHs with nans from two sources
    # 1. Some units' beta coefficients are nan (due to bad fitting), which is handled here
    # 2. Some grouped PSTH are nan (due to missing groups in PSTH); don't need to handle
    # nan_groups = np.isnan(psth).all(axis=2).all(axis=0)
    psth_reshaped = psth.reshape(psth.shape[0], -1)   # From (n_neuron, n_group, n_time) to (n_neuron, n_group * n_time)
    psth_sem_reshaped = psth_sem.reshape(psth_sem.shape[0], -1)
    nan_units = np.isnan(coding_direction)
    
    psth_reshaped_valid = psth_reshaped[~nan_units]
    psth_sem_reshaped_valid = psth_sem_reshaped[~nan_units]
    
    coding_direction_valid = coding_direction[~nan_units]  # Remove nan units (temporary fix)
    coding_direction_valid = coding_direction_valid / np.sqrt(np.sum(coding_direction_valid**2)) # Renormalize
    
    # Do projection
    psth_proj = (psth_reshaped_valid.T @ coding_direction_valid).reshape(psth.shape[1:])
    
    if not if_error_bar:
        return psth_proj, None
    
    # Compute 95% CI from psth_sem (assuming uncertainties all come from psth, not betas; also, I didn't take care of the trial number, since Var = SEM^2 * trial_num)
    psth_proj_sem = np.sqrt((psth_sem_reshaped_valid.T**2 @ coding_direction_valid**2)).reshape(psth_sem.shape[1:])
    psth_proj_95CI = 1.96 * psth_proj_sem
        
    return psth_proj, psth_proj_95CI


def plot_psth_proj_on_CDs(
                     model, psth_align_to,
                     paras, psth_grouped_bys,
                    #  combine_araes=True,
                    if_error_bar=False,
                    lr_or_ic='Left and Right',
                    ):

    # Retrieve unit_keys from dataset
    df_unit_keys = ds_psth[primary_keys].to_dataframe().reset_index()
    df_filtered_unit_with_aoi = st.session_state.df_unit_filtered[primary_keys + ['area_of_interest'] + ['ccf_z']]
    
    # Add aoi to df_unit_keys; right join to apply the filtering
    # Because ds_psth and ds_linear_fit_over_time share the same unit_ind, we can use the same unit_ind to filter
    df_unit_keys_filtered_and_with_aoi = df_unit_keys.merge(df_filtered_unit_with_aoi, on=primary_keys, how='right') 
    unit_ind_filtered = df_unit_keys_filtered_and_with_aoi.unit_ind
    
    # Get hemisphere info
    right_unit_to_flip = df_unit_keys_filtered_and_with_aoi[df_unit_keys_filtered_and_with_aoi.ccf_z >= 5700].index.values
    lr_or_ic_text = {'Ipsi and Contra': '(I/C)', 'Left and Right': '(L/R)'}[lr_or_ic]
    
    st.markdown(f'##### (N = {len(df_unit_keys_filtered_and_with_aoi)} filtered on the side bar)')
    st.markdown(f'##### PSTH bin size = {ds_psth.bin_size:.2g} s, '
                f'smoothed by half Gaussian kernel with $\sigma$ = {ds_psth.psth_aligned_to_Choice_grouped_by_choice_and_reward.smooth_sigma:.2g} s')

    if len(unit_ind_filtered) == 0:
        st.write('No units selected!')
        return
    
    coding_directions = []
    
    for para in paras:  # Iterate over rows
        
        cols = st.columns([0.6] + [1] * 4, gap="medium")    # To fix column width
        
        # Select time epoch for computing coding direction
        cols[0].markdown(f'''### {para} {lr_or_ic_text if para == 'dQ' or 'choice' in para else ''}''')
        cols[0].markdown(f'###### Time epoch for CD:')
        coding_direction_align_to = cols[0].selectbox('align to', 
                                                      align_tos,
                                                      align_tos.index(
                                                          coding_direction_beta_aver_epoch[para]['align_to']),
                                                      key=f'coding_direction_align_to_{para}')
        cc = cols[0].columns([1, 1])
        win_start = cc[0].text_input('start (sec)', 
                                     value=coding_direction_beta_aver_epoch[para]['win'][0],
                                     key=f'coding_direction_win_start_{para}',
                                     )
        win_end = cc[1].text_input('end (sec)',
                                   value = coding_direction_beta_aver_epoch[para]['win'][1],
                                   key=f'coding_direction_win_end_{para}',
                                   )       
        
        # Compute coding direction (essentially normalized betas)
        # Note that coding direction has independent align_to from psth
        coding_direction = _get_coding_direction(model=model, 
                                                 para=para, 
                                                 align_to=coding_direction_align_to, 
                                                 beta_aver_epoch=[win_start, win_end],
                                                 select_units=unit_ind_filtered.values,
                                                 )
        
        # Flip lr to ipsi/contra if needed
        if lr_or_ic == 'Ipsi and Contra' and para in ['dQ', 'choice_this', 'choice_next']:
            coding_direction[right_unit_to_flip] = -coding_direction[right_unit_to_flip]
        
        # Cache CDs
        coding_directions.append(coding_direction)
                
        # Plot units contribution to coding direction
        fig = px.bar(np.sort(coding_direction))
        fig.update_layout(showlegend=False, height=200, 
                            yaxis=dict(title='sorted contribution to CD'),
                            xaxis=dict(title='units'),
                            margin=dict(l=0, r=0, t=0, b=0))
        cols[0].plotly_chart(fig, use_container_width=True)
        
        # Plot PSTH projected on CD
        for i, psth_grouped_by in enumerate(psth_grouped_bys):
    
            # Retrieve psths
            psth_name = f'''psth_aligned_to_''' +\
                        f'''{psth_align_mapping[psth_align_to]}_''' +\
                        f'''grouped_by_{psth_grouped_by}'''

            psth, psth_sem, psth_t, psth_group_names, psth_plot_specs = _get_psth(psth_name=psth_name, 
                                                                            align_to=psth_align_to, 
                                                                            psth_grouped_by=psth_grouped_by,
                                                                            select_units=unit_ind_filtered.values,
                                                                            lr_or_ic=lr_or_ic,
                                                                            right_unit_to_flip=right_unit_to_flip,
                                                                            )
                        
            # Compute projection
            psth_proj, psth_proj_95CI = compute_psth_proj_on_CD(psth=psth, 
                                                                psth_sem=psth_sem,
                                                                coding_direction=coding_direction, 
                                                                if_error_bar=if_error_bar
                                                                )
            
            # Do plotting
            fig = go.Figure()            
            for i_group in range(psth_proj.shape[0]):
                
                if not if_error_bar:
                    fig.add_trace(go.Scatter(x=psth_t, 
                                            y=psth_proj[i_group, :],
                                            mode='lines', 
                                            name=psth_group_names[i_group],
                                            **eval(psth_plot_specs[i_group]),
                                            ),
                                )
                else:
                    # Hack of the dash and line width
                    line_spec = eval(psth_plot_specs[i_group])
                    if 'line_dash' in line_spec:
                        line_dash = line_spec['line_dash']
                        line_width = 2 if line_dash == 'dot' else 3
                    else:
                        line_dash = 'solid'
                        line_width = 3
                    
                    add_plotly_errorbar(x=pd.Series(psth_t), 
                                        y=pd.Series(psth_proj[i_group, :]),
                                        err=pd.Series(psth_proj_95CI[i_group, :]), 
                                        color=eval(psth_plot_specs[i_group])['marker_color'],
                                        line_dash=line_dash,
                                        line_width=line_width,
                                        name=psth_group_names[i_group],
                                        mode='lines',
                                        fig=fig, alpha=0.2, 
                                        )            
            
            fig.update_layout(
                            font_size=17, 
                            hovermode='closest',
                            xaxis=dict(title=f'Time to {psth_align_to} (sec)',
                                        title_font_size=20,
                                        tickfont_size=20),
                            yaxis=dict(visible=False),
                            legend=dict(
                                yanchor="top", y=1.3,
                                xanchor="left", x=0,
                                orientation="h",
                                font_size=15,
                            ),
                            )

            
            fig.for_each_xaxis(lambda x: x.update(showgrid=True))
            
            # Add indicators for other time points
            for other_name, other_time in plot_settings[psth_align_to]['others'].items():
                fig.add_vline(x=other_time, line_color='black', line_dash='dash',
                              name=psth_align_to)
                
            fig.add_vline(x=0, line_color='black', line_dash='solid',
                          name=psth_align_to)

            
            # Increase line width for all scatter traces
            # for trace in fig.data:
            #     if trace.type == 'scatter' and 'line' in dir(trace):
            #         if trace.line.width is None:
            #             trace.line.width = 2
            #         else:
            #             trace.line.width = trace.line.width + 1
        
            with cols[i+1]:
                st.plotly_chart(fig,
                                use_container_width=True,
                               )
    
    # Add corrleation matrix of CD
    
    def plot_corr_matrix(z):
        np.fill_diagonal(z, np.nan)  # Remove diagonal
        # z[np.tril_indices(z.shape[0], 0)] = np.nan  # Remove upper triangle
        
        max_val = np.nanmax(np.abs(z - 90))
        # fig = px.imshow(z,
        #                 zmin=0, #90-max_val*1.1,  # Keep 90 deg in the middle
        #                 zmax=90+max_val*1.1,
        #                 x=paras,
        #                 y=paras,
        #                 color_continuous_scale='RdBu_r', 
        #                 origin='upper')
        
        fig = go.Figure()
        fig.add_trace(go.Heatmap(z=z,
                                 zmin=90-max_val*1.1,  # Keep 90 deg in the middle
                                 zmax=90+max_val*1.1,
                                 zauto=False,
                                 colorscale='RdBu',
                                 x=paras,
                                 y=paras,
                                 )
                      )
        
        fig.update_layout(**plotly_font(17),
                          **plotly_square,
                          yaxis_autorange='reversed',
                          xaxis_side='top',
                          plot_bgcolor='black',
                        )
        fig.for_each_xaxis(lambda x: x.update(showgrid=False))
        fig.for_each_yaxis(lambda x: x.update(showgrid=False))
        
        return fig
    
    # # Pearson correlation coef
    # corr_matrix = np.corrcoef(np.array(coding_directions))
    # fig_corr = plot_corr_matrix(corr_matrix)
    
    # cos(theta)
    cos_thetas = np.array([np.dot(cd1, cd2) for cd1 in coding_directions for cd2 in coding_directions]
                          ).reshape(len(paras), len(paras))
    thetas = np.arccos(cos_thetas) / np.pi * 180
    fig_theta = plot_corr_matrix(thetas)
    
    cols = st.columns([1, 1, 1])
    # with cols[0]:
    #     st.markdown('#### Correlation matrix of coding directions')
    #     st.plotly_chart(fig_corr)
    with cols[0]:
        st.markdown('#### Angles between coding directions')
        plotly_events(fig_theta)
    

if __name__ == '__main__':

    if 'df' not in st.session_state: 
        init()
        
    with st.sidebar:    
        add_unit_filter()

    
    # --- 1. Fetch zarr ---
    try:   # Try local first (if in CO)
        assert 1
        zarr_store = '/root/capsule/data/datajoint_psth_combined/all_linear_fit_over_time.zarr'
        ds_linear_fit_over_time = xr.open_zarr(zarr_store, consolidated=True)
        
        zarr_store = '/root/capsule/data/datajoint_psth_combined/all_psth_over_time.zarr'
        ds_psth = xr.open_zarr(zarr_store, consolidated=True)
        
    except:  # Else, from s3
        fs = s3fs.S3FileSystem(anon=False)
        
        s3_path = f's3://aind-behavior-data/Han/ephys/export/psth/datajoint_psth_combined/all_linear_fit_over_time.zarr'
        zarr_store = s3fs.S3Map(root=s3_path, s3=fs, check=True)
        ds_linear_fit_over_time = xr.open_zarr(zarr_store, consolidated=True)

        s3_path = f's3://aind-behavior-data/Han/ephys/export/psth/datajoint_psth_combined/all_psth_over_time.zarr'
        zarr_store = s3fs.S3Map(root=s3_path, s3=fs, check=True)
        ds_psth = xr.open_zarr(zarr_store, consolidated=True)

    
    align_tos = ds_linear_fit_over_time.align_tos
    models = ds_linear_fit_over_time.linear_models
    
    win_width = ds_linear_fit_over_time['linear_fit_para_stats_aligned_to_choice'].win_width
    win_step = ds_linear_fit_over_time['linear_fit_t_center_aligned_to_choice'][1].values\
             - ds_linear_fit_over_time['linear_fit_t_center_aligned_to_choice'][0].values
    
    plotly_square = dict(xaxis=dict(scaleanchor="y", constrain="domain"),   # Make it square
                         yaxis=dict(constrain="domain"))

    st.markdown('### Select a tab here 👇')
    chosen_id = stx.tab_bar(data=[
                                stx.TabBarItemData(id="tab1", title="1. Fitting stats", description=""),
                                stx.TabBarItemData(id="tab2", title="2. Coding directions", description=""),
                                stx.TabBarItemData(id="tab3", title="3. Projection on CDs", description=""),
                        ], 
                        default="tab3")
    
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
        
        colss = cols[0].columns([1, 1, 1])    
        if_95_CI = colss[0].checkbox('95% CI', True)
        if_sync_y = colss[1].checkbox('Sync y-axis', True)
        if_use_ccf_color = colss[2].checkbox('Use CCF color', True)
        
        # --- 3. Plot ---
        if selected_model and selected_paras and selected_align_to:
            
            fig = plot_linear_fitting_over_time(ds=ds_linear_fit_over_time, 
                                                model=selected_model,
                                                paras=selected_paras,
                                                para_stat=selected_para_stat,
                                                aggr_func=selected_agg_func,
                                                align_tos=selected_align_to,
                                                if_sync_y=if_sync_y,
                                                if_95_CI=if_95_CI,
                                                if_use_ccf_color=if_use_ccf_color,
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
        down_sample_t = cols[1].columns([1, 3])[0].slider('Down sample',
                                       1, 5, 1, 1)
        
        align_to = selected_align_to[0]        

        plot_beta_auto_corr(ds=ds_linear_fit_over_time, 
                                  model=selected_model, 
                                  paras=selected_paras, 
                                  align_tos=selected_align_to,
                                  down_sample_t=down_sample_t)
    
    elif chosen_id == 'tab3':
                
        cols = st.columns([0.5, 1, 1, 0.5])
        selected_model = cols[0].selectbox('Linear model',
                                                list(models.keys()), 
                                                1,
                                            )
                        
        available_paras_this_model = models[selected_model]
        # if_combine_araes = cols[0].checkbox('Combine areas', True)
        if_error_bar = cols[0].checkbox('Show 95% CI', True)
        
        selected_paras = cols[1].multiselect('Coding directions',
                                            coding_direction_beta_aver_epoch.keys(), 
                                            coding_direction_beta_aver_epoch.keys()
                                            )
        
        selected_lr_or_ic = cols[1].selectbox('L/R or Ipsi/Contra',
                                              ['Left and Right', 
                                               'Ipsi and Contra'], 
                                              0
                                            )
        
        psth_grouped_bys = list(ds_psth.psth_setting_grouped_bys.keys())
        selected_grouped_bys = cols[2].multiselect('PSTH grouped bys',
                                        psth_grouped_bys, 
                                        psth_grouped_bys
                                        )
        psth_align_to = cols[3].selectbox('PSTH align to', 
                                        align_tos,
                                        1)
        
        psth_align_mapping = {'choice': 'Choice', 
                              'iti_start': 'ITI_start', 'go_cue': 'Go_cue'}
        
        plot_psth_proj_on_CDs(
                            model=selected_model, 
                            psth_align_to=psth_align_to,
                            paras=selected_paras,
                            psth_grouped_bys=selected_grouped_bys,
                            # combine_araes=if_combine_araes
                            if_error_bar=if_error_bar,
                            lr_or_ic = selected_lr_or_ic,
                            )        

                 
    if if_debug:
        p.stop()