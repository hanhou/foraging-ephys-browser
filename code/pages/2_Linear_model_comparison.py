import streamlit as st
import numpy as np
import pandas as pd

import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

from streamlit_plotly_events import plotly_events
import extra_streamlit_components as stx

from util import *
from util.selectors import (add_unit_filter, select_period, select_model, select_para)
from Home import init, _to_theta_r, select_t_sign_level, t_to_p, p_to_t




@st.cache_data(max_entries=100)
def plot_model_comparison():
    df_period_linear_fit_melt = df_period_linear_fit_filtered.iloc[:, 
                                                                   df_period_linear_fit_filtered.columns.get_level_values(-2)=='rel_bic'
                                                            ].stack(level=[0, 1]
                                                                    ).droplevel(axis=1, level=1
                                                                                ).reset_index()
    models = [m for m in all_models if 'rew' in m]     # Manual filtering                                                                  
    fig = px.box(df_period_linear_fit_melt.query('multi_linear_model in @models'
                                                 ).replace(model_name_mapper),
                 x='period', y='rel_bic', 
                 color='multi_linear_model', 
                 color_discrete_sequence=list(model_color_mapper.values()),
                 category_orders={"multi_linear_model": model_name_mapper.values()}, 
                 )
    
    return fig

@st.cache_data(max_entries=100, show_spinner=False)
def plot_t_distribution(df_period_linear_fit, periods, paras, to_compare='models'):

    fig = make_subplots(rows=len(paras), cols=len(periods), 
                        subplot_titles=[period_name_mapper[period] for period in periods],
                        shared_yaxes=True,
                        shared_xaxes=True,
                        vertical_spacing=0.02,
                        x_title='Threshold for abs(t)',
                        )
    
    progress_bar = st.columns([1, 15])[0].progress(0, text='0%')

    for row, para in enumerate(paras):
        for col, period in enumerate(periods):
            df = df_period_linear_fit.loc[:, (period, 
                                              df_period_linear_fit.columns.get_level_values('multi_linear_model'), 
                                              "t", 
                                              para)
                                         ].droplevel(axis=1, level=0
                                                     ).stack([0, 2]
                                                             ).reset_index(
                                                                 ).set_index(unit_key_names)
            df['abs(t)'] = df.t.abs()
            
            if to_compare == 'models':
                models = [m for m in all_models if 'dQ' in m]
                model_highlight = ('dQ, sumQ, rew, chQ', 'dQ, sumQ, rew, chQ, C*2, R*5, t')

                for i, model in enumerate(models):
                    hist, x = np.histogram(df.query(f'multi_linear_model == "{model}"')['abs(t)'], 
                                        bins=100)
                    n = np.sum(hist)
                    sign_ratio = 1 - np.cumsum(hist) / n if n > 0 else np.full(hist.shape, np.nan)
                    fig.add_trace(go.Scattergl(x=(x[:-1] + x[1:])/2, 
                                            y=sign_ratio,
                                            mode='lines',
                                            line=dict(color=model_color_mapper[model],
                                                      width=5 if model in model_highlight else 3),
                                            name=model_name_mapper[model],
                                            legendgroup=model,
                                            showlegend=col==0 and row ==0,
                                            hovertemplate=
                                                '%s<br>' % (model) +
                                                '%{y:%2.1f} units, t > %{x:.2f}<br><extra></extra>',
                                            visible=True if model in model_highlight else 'legendonly',
                                            ),
                                row=row+1, col=col+1)
            elif to_compare == 'areas':
                for area, color in st.session_state.aoi_color_mapping.items():
                    hist, x = np.histogram(df.query(f'area_of_interest == "{area}"')['abs(t)'], bins=100)
                    n = np.sum(hist)
                    sign_ratio = 1 - np.cumsum(hist) / n if n > 0 else np.full_like(hist, np.nan)
                    fig.add_trace(go.Scattergl(x=(x[:-1] + x[1:])/2, 
                                                y=sign_ratio,
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
                    
            # t-value
            fig.add_trace(go.Scatter(x=np.linspace(0, 5, 500),
                                    y=t_to_p(np.linspace(0, 5, 500)),
                                    mode='lines',
                                    name='Type I error',
                                    legendgroup='err',
                                    showlegend=col==0 and row ==0,
                                    line=dict(color='gray', dash='dash'),
                                    hovertemplate='t > %{x:.2f}, p < %{y:.3f}<br><extra></extra>',
                                    ),
                        row=row+1, col=col+1)

            fig.add_vline(x=2.0, line_color='gray', line_dash='dash',
                        row=row+1, col=col+1)
            
            finished = (row * len(periods) + col + 1) / (len(periods) * len(paras))
            progress_bar.progress(finished, text=f'{finished:.0%}')


    for row, para in enumerate(paras):
        fig['layout'][f'yaxis{1 + row * len(periods)}']['title'] = para_name_mapper[para]

    # fig.update_traces(line_width=3)
    fig.update_xaxes(range=[0, 5])
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(width=min(2000, 400 + 270 * len(periods)), 
                      height=200 + 240 * len(paras),
                     font_size=17, hovermode='closest',
                     title= f'(Total number of units = {len(df_period_linear_fit)})',
                     title_x=0.01,
                     )
    fig.update_annotations(font_size=20)
        
    return fig

def _sig_proportion(ts, t_sign_level):
    prop = np.sum(np.abs(ts) >= t_sign_level) / len(ts)
    ci_95 = 1.96 * np.sqrt(prop * (1 - prop) / len(ts))
    return prop * 100, ci_95 * 100, len(ts)


def plot_unit_sig_prop_bar(aois, period, t_sign_level, rpe_or_reward_and_Q='rpe'):
    p_value = t_to_p(t_sign_level)
    
    if rpe_or_reward_and_Q == 'RPE':  # Whether separate reward and Q
        model_groups = {('dQ, sumQ, rpe', 'contraQ, ipsiQ, rpe'): ['simple model',   # Name
                                                                [p for p in sig_prop_vars if 'action_value' in p or p in ['rpe']], # Para to include
                                                                dict(pattern_shape='/', pattern_fillmode="replace") # Setting
                                                                ],
                         ('dQ, sumQ, rpe, C*2, R*5, t', 'contraQ, ipsiQ, rpe, C*2, R*5, t'): ['full model',
                                                                                            [p for p in sig_prop_vars if 'action_value' in p or p in 
                                                                                            ['rpe', 'choice_ic', 'choice_ic_next', 'trial_normalized', 'firing_1_back']],
                                                                                            dict()],
        }
    else:
        model_groups = {('dQ, sumQ, rew, chQ', 'contraQ, ipsiQ, rew, chQ'): ['simple model',   # Name
                                                                [p for p in sig_prop_vars if 'action_value' in p or p in ['reward', 'chosen_value']], # Para to include
                                                                dict(pattern_shape='/', pattern_fillmode="replace") # Setting
                                                                ],
                        ('dQ, sumQ, rew, chQ, C*2, R*5, t', 'contraQ, ipsiQ, rew, chQ, C*2, R*5, t'): ['full model',
                                                                                            [p for p in sig_prop_vars if 'action_value' in p or p in 
                                                                                            ['reward', 'chosen_value', 'choice_ic', 'choice_ic_next', 'trial_normalized', 'firing_1_back']],
                                                                                            dict()],
                        }   
    
    fig = go.Figure()
    
    for model_group, model_group_setting in model_groups.items():
        paras = model_group_setting[1]
        for para in paras:
            color = sig_prop_color_mapping[para]
            prop_ci = df_period_linear_fit_filtered.loc[:, (period, model_group, 't', para)
                                                  ].groupby('area_of_interest'
                                                            ).agg(lambda x: _sig_proportion(x, t_sign_level))
            
            # filtered_aoi = [aoi for aoi in st.session_state.df['aoi'].index if aoi in prop_ci.index and aoi in aois] # Sort according to st.session_state.df['aoi'].index
            filtered_aoi = [aoi for aoi in aois if aoi in prop_ci.index and aoi in aois]

            prop_ci = prop_ci.reindex(filtered_aoi)
            prop = [x[0] for x in prop_ci.values[:, 0]] 
            err = [x[1] for x in prop_ci.values[:, 0]] 
            ns =  [x[2] for x in prop_ci.values[:, 0]] 

            fig.add_trace(go.Bar( 
                                name=f'{para_name_mapper[para]} ({model_group_setting[0]})',
                                x=filtered_aoi,
                                y=prop,
                                error_y=dict(type='data', array=err, thickness=1),
                                hovertemplate='%%{x}, %s, %s' % (para_name_mapper[para], period) + 
                                            '<br>%{y:.1f} ± %{customdata[0]:.1f} % (95% CI)' + 
                                            '<br>n = %{customdata[1]} <extra></extra>',
                                customdata=np.stack((err, ns), axis=-1),
                                marker=dict(
                                            color='white' if model_group_setting[0] == 'simple model' else color,
                                            line_color=color,
                                            line_width=1,
                                            **model_group_setting[2],
                                            ),
                                ))
        
    fig.add_hline(y=p_value * 100,
                  line_color='black', line_dash='dash')
    fig.add_hline(y=100,  
                  line_color='black', line_dash='dash')

    
    fig.update_layout(barmode='group', 
                      height=500,
                      yaxis_title='% sig. units (+/- 95% CI)',
                      font=dict(size=20),
                      hovermode='closest',
                      title=f'p < {p_value:.2g}',
                      yaxis_range=[0, 105],
                      **plotly_font(20),  # Have to use this because we have to use plotly_chart to keep the patterns...  
                      title_font_size=25,
                      legend=dict(
                                yanchor="top", y=1.2,
                                xanchor="right", x=1,
                                orientation="h",
                            )
                      )    
    return fig

def _pure_proportion(x):
    prop = sum(x) / len(x)
    ci_95 = 1.96 * np.sqrt(prop * (1 - prop) / len(x))
    return prop * 100, ci_95 * 100, len(x)

def plot_unit_pure_sig_prop_bar(aois, period, t_sign_level, model='dQ, sumQ, rew, chQ'):
    p_value = t_to_p(t_sign_level)

    model_groups = {(model): ['simple model',   # Name
                                polar_classifiers[model][1].keys(), # Para to include
                                dict(pattern_shape='/', pattern_fillmode="replace") # Setting
                                ],
                    (model+', R*5, t') if 'dQ' in model else (model+', C*2, R*5, t'): ['full model',
                                                                                        polar_classifiers[model][1].keys(),
                                                                                        dict()],
                    }   
    
    fig = go.Figure()
    
    for model_group, model_group_setting in model_groups.items():
        
        for unit_class, ranges in polar_classifiers[model][1].items():
            color = pure_unit_color_mapping[unit_class]         
            prop_ci = df_period_linear_fit_filtered[period, model_group, f'{unit_class}', ''].groupby('area_of_interest').apply(_pure_proportion)  
                        
            # filtered_aoi = [aoi for aoi in st.session_state.df['aoi'].index if aoi in prop_ci.index and aoi in aois] # Sort according to st.session_state.df['aoi'].index
            filtered_aoi = [aoi for aoi in aois if aoi in prop_ci.index and aoi in aois]

            prop_ci = prop_ci.reindex(filtered_aoi)
            prop = [x[0] for x in prop_ci]
            err = [x[1] for x in prop_ci]
            ns =  [x[2] for x in prop_ci] 

            fig.add_trace(go.Bar(
                                name=f'{unit_class} ({model_group_setting[0]})',
                                x=prop_ci.index, 
                                y=prop,
                                error_y=dict(type='data', array=err),
                                marker=dict(
                                            color='white' if model_group_setting[0] == 'simple model' else color,
                                            line_color=color,
                                            line_width=1,
                                            **model_group_setting[2],
                                            ), 
                                hovertemplate='%%{x}, %s' % (unit_class) + 
                                            '<br>%{y:.1f} ± %{customdata[0]:.1f} % (95% CI)' + 
                                            '<br>n = %{customdata[1]} <extra></extra>',
                                customdata=np.stack((err, ns), axis=-1),
                                ))
        
    fig.add_hline(y=p_value * 100 / 4,  # Divided by 4 types 
                  line_color='black', line_dash='dash')
    fig.add_hline(y=100,  # Divided by 4 types 
                  line_color='black', line_dash='dash')
    
    fig.update_layout(barmode='group', 
                      height=700,
                      yaxis_title='% sig. units (+/- 95% CI)',
                      font=dict(size=20),
                      hovermode='closest',
                      title=f'Polar classification from ({model[:-5]}), p < {p_value:.2g}',
                      title_font_size=25,
                      yaxis_range=[0, 105],
                      **plotly_font(20),  # Have to use this because we have to use plotly_chart to keep the patterns...  
                      legend=dict(
                                yanchor="top", y=0.9,
                                xanchor="right", x=1,
                                orientation="h",
                            )
                      )    
    return fig


def plot_unit_class_scatter(period, model='dQ, sumQ, rew, chQ'):

    x_name, y_name = polar_classifiers[model][0].values()
    models = {(model): 'simple model', (model+', R*5, t') if 'dQ' in model else (model+', C*2, R*5, t'): 'full model'}  

    # fig = make_subplots(rows=1, cols=2, column_titles=list(model_groups.keys()))
    
    figs = []
    
    for i, (model, descr) in enumerate(models.items()):
        fig = go.Figure()
        for unit_class, color in pure_unit_color_mapping.items():
            this = df_period_linear_fit_filtered[period, (model), f'{unit_class}', ''].astype(bool)
                
            fig.add_trace(go.Scattergl(x=df_period_linear_fit_filtered[period, (model), 't', x_name][this], 
                                       y=df_period_linear_fit_filtered[period, (model), 't', y_name][this], 
                                       mode='markers',
                                       marker=dict(symbol='circle', size=7, opacity=0.3, 
                                                line=dict(color=color, width=1.5),
                                                color=color if 'R*5' in (model) else 'white'), 
                                       name=f'{unit_class}'
                                      ), 
                        #   row=1, col=i+1
                          )
        
        fig.update_layout(width=700, height=700, font=dict(size=20),
                        xaxis_title=x_name, yaxis_title=y_name, **plotly_font(20),
                        xaxis_range=[-20, 20], yaxis_range=[-20, 20],
                        hovermode='closest',
                        legend=dict(yanchor="bottom", y=0, xanchor="right", x=1, orientation="v", font_size=15),
                        title=f'{descr} ({model_name_mapper[model]})',
                        title_font_size=15,
                        )
        fig.update_xaxes(scaleanchor = "y", scaleratio = 1)
        figs.append(fig)
            
    return figs


if __name__ == '__main__':

    if 'df' not in st.session_state: 
        init()
    
    with st.sidebar:    
        add_unit_filter()
            
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

    # --- Model comparison ---
    st.markdown('### Select a tab here 👇')
    chosen_id = stx.tab_bar(data=[
                                stx.TabBarItemData(id="tab1", title="1. Model comparison", description=""),
                                stx.TabBarItemData(id="tab2", title="2. Distribution of t-values, compare models", description=""),
                                stx.TabBarItemData(id="tab3", title="3. Distribution of t-values, compare areas", description=""),
                                stx.TabBarItemData(id="tab4", title="4. Proportion of significant units", description=""),
                                ], 
                            default="tab2")

    if chosen_id == 'tab1':
        st.markdown('##### :red[Model comparison, all units]')
        fig = plot_model_comparison()

        fig.update_layout(width=2000, height=700, 
                        #   title='Model comparison for unit period fitting',
                        xaxis_title='Period',
                        yaxis_title='Model BIC',
                        **plotly_font(20)
                        )

        fig.update_xaxes(categoryorder='array', categoryarray=all_models)
        st.plotly_chart(fig, use_container_width=True)

    elif chosen_id == 'tab2':

        # --- t-distribution, compare models ---
        st.markdown('#### :red[Proportion of significant units V.S. t-value threshold (compare models)]')

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
        aois = cols[0].multiselect('Areas to include', st.session_state.aoi_color_mapping.keys(), st.session_state.aoi_color_mapping)
   
        _, paras = select_para(default_paras=[p for p in para_name_mapper if para_name_mapper[p] in 
                                              ['dQ', 'sumQ', 'reward', 'chosenQ', 'choice (this)', 'choice (next)', 'trial number', 'firing 1 back']], 
                               col=cols[1])
        _, periods = select_period(col=cols[2])
        
        if aois and paras and periods:
            df_period_linear_fit = df_period_linear_fit_filtered.query('area_of_interest in @aois')
            fig = plot_t_distribution(df_period_linear_fit=df_period_linear_fit, 
                                    periods=periods, 
                                    paras=paras,
                                    to_compare='models',
                                    )

            plotly_events(fig, override_height=fig.layout.height*1.1, override_width=fig.layout.width, click_event=False)

    elif chosen_id == 'tab3':
        
        # --- t-distribution, compare areas ---
        st.markdown('#### :red[Proportion of significant units V.S. t-value threshold (compare areas)]')

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
        
        _, model = select_model(col=cols[0])
        
        df_this_model = df_period_linear_fit_filtered.iloc[:, df_period_linear_fit_filtered.columns.get_level_values('multi_linear_model') == model]

        available_paras_this_model = [p for p in para_name_mapper if p in df_this_model.columns.get_level_values('var_name').unique()]
        _, paras = select_para(available_paras=available_paras_this_model, 
                               default_paras=[p for p in available_paras_this_model if 'action_value' in p 
                                              or p in ['reward', 'chosen_value', 'choice_ic', 'choice_ic_next', 'trial_normalized', 'firing_1_back']],
                               col=cols[1])
                
        _, periods = select_period(col=cols[2])

        if paras and periods:
            fig = plot_t_distribution(df_period_linear_fit=df_this_model, 
                                    to_compare='areas',
                                    periods=periods, 
                                    paras=paras)

            plotly_events(fig, override_height=fig.layout.height*1.1, override_width=fig.layout.width, click_event=False)

    elif chosen_id == 'tab4':
        with st.sidebar:
            with st.expander('t-value threshold', expanded=True):
                t_sign_level = select_t_sign_level()
        
        cols = st.columns([1, 2, 1])
        period = cols[0].selectbox('period', 
                                   period_name_mapper.values(), 
                                   list(period_name_mapper.keys()).index('iti_all'))
        
        _period = [p for p in period_name_mapper if period_name_mapper[p] == period][0]
        aois = cols[1].multiselect('Areas to include', st.session_state.aoi_color_mapping.keys(), st.session_state.aoi_color_mapping)
        
        # Whether use rpe or reward + chosenQ
        rpe_or_reward_and_Q = cols[2].selectbox('Use reward + chosenQ or RPE', ['reward + chosenQ', 'RPE'])
        
        # -- bar plot of significant units --
        with st.expander('Bar plot of significant units', expanded=True):
            fig = plot_unit_sig_prop_bar(period=_period,
                                        aois=aois,
                                        t_sign_level=st.session_state['t_sign_level'],
                                        rpe_or_reward_and_Q=rpe_or_reward_and_Q)
            
            st.plotly_chart(fig, use_container_width=True) # Only plotly_chart keeps the bar plot pattern
        
        # -- bar plot of pure units --
        with st.expander('Bar plot of pure units', expanded=True):
            model_name = st.columns([1, 5])[0].selectbox('Model for polar classification', ['dQ + sumQ + ...', 'contraQ + ipsiQ + ...'], 0)
            
            if rpe_or_reward_and_Q == 'reward + chosenQ':
                model = {'dQ + sumQ + ...': 'dQ, sumQ, rew, chQ, C*2', 'contraQ + ipsiQ + ...': 'contraQ, ipsiQ, rew, chQ'}[model_name]
            else:
                model = {'dQ + sumQ + ...': 'dQ, sumQ, rpe, C*2', 'contraQ + ipsiQ + ...': 'contraQ, ipsiQ, rpe'}[model_name]
                
            fig = plot_unit_pure_sig_prop_bar(period=_period,
                                            aois=aois,
                                            t_sign_level=st.session_state['t_sign_level'],
                                            model=model,
                                            )        
            st.plotly_chart(fig, use_container_width=True) # Only plotly_chart keeps the bar plot pattern

        # -- illustrate polar classfier --
        with st.expander('Polar classifier', expanded=True):
            cols = st.columns([1, 1, 1])
            figs = plot_unit_class_scatter(period=_period, model=model)
            for i, fig in enumerate(figs):
                with cols[i]:
                    plotly_events(fig, override_height=fig.layout.height*1.1, override_width=fig.layout.width, click_event=False)