import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm

import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

from streamlit_plotly_events import plotly_events
import extra_streamlit_components as stx

from Home import add_unit_filter, init

if 'df' not in st.session_state: 
    init()
    
with st.sidebar:    
    add_unit_filter()
    
# Prepare df
df_period_linear_fit_all = st.session_state.df['df_period_linear_fit_all']
unit_key_names = ['subject_id', 'session', 'insertion_number', 'unit']
# Filter df and add area_of_interest to the index
df_period_linear_fit_all = df_period_linear_fit_all.loc[st.session_state.df_unit_filtered.set_index(unit_key_names).index, :]
df_aoi_filtered = st.session_state.df['df_ephys_units'].set_index(unit_key_names).loc[df_period_linear_fit_all.index, :]
aoi_index = df_aoi_filtered.reset_index().set_index(unit_key_names + ['area_of_interest'])
df_period_linear_fit_all.index = aoi_index.index

plotly_font = lambda x: dict(xaxis_tickfont_size=x,
                            xaxis_title_font_size=x,
                            yaxis_tickfont_size=x,
                            yaxis_title_font_size=x,
                            legend_font_size=x,
                            legend_title_font_size=x,)

all_models = ['dQ, sumQ, rpe', 
               'dQ, sumQ, rpe, C*2', 
               'dQ, sumQ, rpe, C*2, t', 
               'dQ, sumQ, rpe, C*2, R*1', 
               'dQ, sumQ, rpe, C*2, R*1, t',
               'dQ, sumQ, rpe, C*2, R*5, t',
               'dQ, sumQ, rpe, C*2, R*10, t',
               'contraQ, ipsiQ, rpe',
               'contraQ, ipsiQ, rpe, C*2, R*5, t']
model_color_map = {model:color for model, color in zip(all_models, px.colors.qualitative.Plotly)}

all_periods = ['before_2', 'delay', 'go_1.2', 'go_to_end', 'iti_all', 'iti_first_2', 'iti_last_2']
period_mapping = {'before_2': 'Before GO (2s)', 'delay': 'Delay (median 60 ms)', 'go_1.2': 'After GO (1.2s)', 'go_to_end': 'GO to END', 
                  'iti_all': 'ITI (all, median 3.95s)', 'iti_first_2': 'ITI (first 2s)', 'iti_last_2': 'ITI (last 2s)'}

para_mapping = {'relative_action_value_ic': 'dQ', 'total_action_value': 'sumQ', 
                'contra_action_value': 'contraQ', 'ipsi_action_value': 'ipsiQ',
                'rpe': 'rpe',
                'choice_ic': 'choice (this)', 'choice_ic_next': 'choice (next)',
                'trial_normalized': 'trial number', 
                **{f'firing_{n}_back': f'firing {n} back' for n in range(1, 11)},
                }
all_paras = [var for var in df_period_linear_fit_all.columns.get_level_values('var_name').unique() if var !='']
all_paras = para_mapping.keys()

t_value = np.linspace(0, 5, 500)
type_1_error = 2 * norm.sf(t_value)

pure_unit_color_mapping =  {'pure_dQ': 'darkviolet',
                            'pure_sumQ': 'deepskyblue',
                            'pure_contraQ': 'darkblue',
                            'pure_ipsiQ': 'darkorange'}
                                
sig_prop_vars = ['relative_action_value_ic', 'total_action_value', 
                'contra_action_value', 'ipsi_action_value',
                'rpe',
                'choice_ic', 'choice_ic_next', 'trial_normalized', 'firing_1_back']
sig_prop_color_mapping = {var: color for var, color in zip(sig_prop_vars, 
                                                           ['darkviolet', 'deepskyblue', 'darkblue', 'darkorange', 'gray'] + px.colors.qualitative.Plotly)}


@st.cache_data(max_entries=100)
def plot_model_comparison():
    df_period_linear_fit_melt = df_period_linear_fit_all.iloc[:, df_period_linear_fit_all.columns.get_level_values(-2)=='rel_bic'
                                                            ].stack(level=[0, 1]
                                                                    ).droplevel(axis=1, level=1
                                                                                ).reset_index()
    fig = px.box(df_period_linear_fit_melt.query('multi_linear_model in @all_models'), x='period', y='rel_bic', 
                 color='multi_linear_model', category_orders={"multi_linear_model": all_models}, color_discrete_map=model_color_map)
    return fig

@st.cache_data(max_entries=100)
def plot_t_distribution(df_period_linear_fit, periods, paras, to_compare='models'):

    fig = make_subplots(rows=len(paras), cols=len(periods), 
                        subplot_titles=[period_mapping[period] for period in periods],
                        shared_yaxes=True,
                        shared_xaxes=True,
                        vertical_spacing=0.02,
                        x_title='abs(t)',
                        )

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
                model_highlight = ('dQ, sumQ, rpe', 'dQ, sumQ, rpe, C*2, R*5, t')

                for i, model in enumerate(models):
                    hist, x = np.histogram(df.query(f'multi_linear_model == "{model}"')['abs(t)'], 
                                        bins=100)
                    n = np.sum(hist)
                    sign_ratio = 1 - np.cumsum(hist) / n if n > 0 else np.full(hist.shape, np.nan)
                    fig.add_trace(go.Scattergl(x=(x[:-1] + x[1:])/2, 
                                            y=sign_ratio,
                                            mode='lines',
                                            line=dict(color=model_color_map[model],
                                                    width=5 if model in model_highlight else 3),
                                            name=model,
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
                    sign_ratio = 1 - np.cumsum(hist) / n if n > 0 else np.nan
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
            fig.add_trace(go.Scatter(x=t_value,
                                    y=type_1_error,
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

    for row, para in enumerate(paras):
        fig['layout'][f'yaxis{1 + row * len(periods)}']['title'] = para_mapping[para]

    # fig.update_traces(line_width=3)
    fig.update_xaxes(range=[0, 5])
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(width=min(1600, 2000/6 *len(periods)), height=200 + 200 * len(paras),
                    font_size=17, hovermode='closest',
                    )
    fig.update_annotations(font_size=20)
        
    return fig

def _sig_proportion(ts, t_sign_level):
    prop = np.sum(np.abs(ts) >= t_sign_level) / len(ts)
    ci_95 = 1.96 * np.sqrt(prop * (1 - prop) / len(ts))
    return prop * 100, ci_95 * 100, len(ts)

def plot_unit_sig_prop_bar(df_unit_filtered, aois, period, t_sign_level):
    p_value = type_1_error[np.searchsorted(t_value, t_sign_level)]
    
    model_groups = {('dQ, sumQ, rpe', 'contraQ, ipsiQ, rpe'): ['naive model',   # Name
                                                               [p for p in sig_prop_vars if 'action_value' in p or p in ['rpe']], # Para to include
                                                               dict(pattern_shape='/', pattern_fillmode="replace") # Setting
                                                               ],
                    ('dQ, sumQ, rpe, C*2, R*5, t', 'contraQ, ipsiQ, rpe, C*2, R*5, t'): ['full model',
                                                                                         [p for p in sig_prop_vars if 'action_value' in p or p in 
                                                                                          ['rpe', 'choice_ic', 'choice_ic_next', 'trial_normalized', 'firing_1_back']],
                                                                                         dict()],
                    }   
    
    fig = go.Figure()
    
    for model_group, model_group_setting in model_groups.items():
        paras = model_group_setting[1]
        for para in paras:
            color = sig_prop_color_mapping[para]
            prop_ci = df_period_linear_fit_all.loc[:, (period, model_group, 't', para)
                                                  ].groupby('area_of_interest'
                                                            ).agg(lambda x: _sig_proportion(x, t_sign_level))
            
            filtered_aoi = [aoi for aoi in aois if aoi in prop_ci.index] # Sort according to st.session_state.df['aoi'].index
            prop_ci = prop_ci.reindex(filtered_aoi)
            prop = [x[0] for x in prop_ci.values[:, 0]] 
            err = [x[1] for x in prop_ci.values[:, 0]] 
            ns =  [x[2] for x in prop_ci.values[:, 0]] 

            fig.add_trace(go.Bar( 
                                name=f'{para_mapping[para]} ({model_group_setting[0]})',
                                x=filtered_aoi,
                                y=prop,
                                error_y=dict(type='data', array=err, thickness=1),
                                hovertemplate='%%{x}, %s, %s' % (para_mapping[para], period) + 
                                            '<br>%{y:.1f} ± %{customdata[0]:.1f} % (95% CI)' + 
                                            '<br>n = %{customdata[1]} <extra></extra>',
                                customdata=np.stack((err, ns), axis=-1),
                                marker=dict(
                                            color='white' if model_group_setting[0] == 'naive model' else color,
                                            line_color=color,
                                            line_width=1,
                                            **model_group_setting[2],
                                            ),
                                ))
        
    fig.add_hline(y=p_value * 100, 
                  line_color='black', line_dash='dash')
    
    fig.update_layout(barmode='group', 
                      height=800,
                      yaxis_title='% sig. units (+/- 95% CI)',
                      font=dict(size=20),
                      hovermode='closest',
                      title=f'p < {p_value:.2g}',
                      yaxis_range=[0, 100],
                      **plotly_font(20),  # Have to use this because we have to use plotly_chart to keep the patterns...  
                      title_font_size=25,
                      legend=dict(
                                yanchor="top", y=1.2,
                                xanchor="right", x=1,
                                orientation="h",
                            )
                      )    
    return fig


# --- Model comparison ---
st.markdown('### Select tab here 👇')
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
    st.plotly_chart(fig)

elif chosen_id == 'tab2':

    # --- t-distribution, compare models ---
    st.markdown('#### :red[Distribution of t-values, compare models]')

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
    paras = cols[1].multiselect('Variables to draw', 
                                [para_mapping[p] for p in all_paras], 
                                [para_mapping[p] for p in ['relative_action_value_ic', 'total_action_value', 'rpe',
                                                           'choice_ic', 'choice_ic_next', 'trial_normalized', 'firing_1_back']])
    periods = cols[2].multiselect('Periods to draw', 
                                  [period_mapping[p] for p in all_periods],
                                  [period_mapping[p] for p in all_periods if p!= 'delay'])

    if aois and paras and periods:
        df_period_linear_fit = df_period_linear_fit_all.query('area_of_interest in @aois')
        st.markdown(f'#### N = {len(df_period_linear_fit)}')
        fig = plot_t_distribution(df_period_linear_fit=df_period_linear_fit, 
                                  periods=[p for p in all_periods if period_mapping[p] in periods], 
                                  paras=[p for p in all_paras if para_mapping[p] in paras],
                                  to_compare='models',
                                  )

        plotly_events(fig, override_height=fig.layout.height*1.1, override_width=fig.layout.width, click_event=False)

elif chosen_id == 'tab3':
    
    # --- t-distribution, compare areas ---
    st.markdown('#### :red[Distribution of t-values, compare areas]')

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
    model = cols[0].selectbox('Model to plot', all_models, all_models.index('dQ, sumQ, rpe, C*2, R*5, t'))
    df_this_model = df_period_linear_fit_all.iloc[:, df_period_linear_fit_all.columns.get_level_values('multi_linear_model') == model]

    availabe_paras_this_model = [p for p in all_paras if p in df_this_model.columns.get_level_values('var_name').unique()]
    paras = cols[1].multiselect('Variables to draw', 
                                [para_mapping[p] for p in availabe_paras_this_model], 
                                [para_mapping[p] for p in availabe_paras_this_model 
                                 if 'action_value' in p 
                                 or p in ['rpe', 'choice_ic', 'choice_ic_next', 'trial_normalized', 'firing_1_back']])
    
    periods = cols[2].multiselect('Periods to draw', 
                                  [period_mapping[p] for p in all_periods],
                                  [period_mapping[p] for p in all_periods if p!= 'delay'])

    if paras and periods:
        fig = plot_t_distribution(df_period_linear_fit=df_this_model, 
                                  to_compare='areas',
                                  periods=[p for p in all_periods if period_mapping[p] in periods], 
                                  paras=[p for p in all_paras if para_mapping[p] in paras])

        plotly_events(fig, override_height=fig.layout.height*1.1, override_width=fig.layout.width, click_event=False)

elif chosen_id == 'tab4':
    cols = st.columns([1, 2, 1])
    period = cols[0].selectbox('period', [period_mapping[p] for p in all_periods], 
                               [period_mapping[p] for p in all_periods].index(period_mapping['iti_all']))
    t_sign_level = cols[0].slider('t value threshold', 1.0, 5.0, 2.57)
    aois = cols[1].multiselect('Areas to include', st.session_state.aoi_color_mapping.keys(), st.session_state.aoi_color_mapping)
    
    fig = plot_unit_sig_prop_bar(st.session_state.df_unit_filtered, 
                                 period=[p for p in all_periods if period_mapping[p] == period],
                                 aois=aois,
                                 t_sign_level=t_sign_level)
    
    st.plotly_chart(fig, use_container_width=True) # Only plotly_chart keeps the bar plot pattern
    # plotly_events(fig, override_height=fig.layout.height*1.1, override_width=fig.layout.width, click_event=False)
    
    
    # paras = cols[1].multiselect('Variables to draw', 
    #                             [para_mapping[p] for p in sig_prop_vars], 
    #                             [para_mapping[p] for p in sig_prop_vars 
    #                              if 'action_value' in p 
    #                              or p in ['rpe']])