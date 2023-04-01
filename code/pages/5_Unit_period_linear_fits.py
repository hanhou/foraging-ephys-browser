import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm

import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

from streamlit_plotly_events import plotly_events
import extra_streamlit_components as stx

from Home import init

if 'df' not in st.session_state: 
    init()
    
# Prepare df
df_period_linear_fit_all = st.session_state.df['df_period_linear_fit_all']


plotly_font = lambda x: dict(xaxis_tickfont_size=x,
                            xaxis_title_font_size=x,
                            yaxis_tickfont_size=x,
                            yaxis_title_font_size=x,
                            legend_font_size=x,
                            legend_title_font_size=x,)

unit_key_names = ['subject_id', 'session', 'insertion_number', 'unit']

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

@st.cache_data(ttl=3600*24)
def plot_model_comparison():
    df_period_linear_fit_melt = df_period_linear_fit_all.iloc[:, df_period_linear_fit_all.columns.get_level_values(-2)=='rel_bic'
                                                            ].stack(level=[0, 1]
                                                                    ).droplevel(axis=1, level=1
                                                                                ).reset_index()
    fig = px.box(df_period_linear_fit_melt.query('multi_linear_model in @all_models'), x='period', y='rel_bic', 
                 color='multi_linear_model', category_orders={"multi_linear_model": all_models}, color_discrete_map=model_color_map)
    return fig

@st.cache_data(ttl=3600*24)
def plot_t_distribution(df_period_linear_fit, periods, paras, to_compare='models'):

    fig = make_subplots(rows=len(paras), cols=len(periods), 
                        subplot_titles=[period_mapping[period] for period in periods],
                        shared_yaxes=True,
                        shared_xaxes=True,
                        vertical_spacing=0.02,
                        x_title='abs(t)',
                        )

    t_value = np.linspace(0, 10, 100)
    type_1_error = 2 * norm.sf(t_value)

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
                    sign_ratio = 1 - np.cumsum(hist) / np.sum(hist)
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
                df_aoi = df.join(st.session_state.df['df_ephys_units'].set_index(unit_key_names)['area_of_interest'], how='left')
                for area, color in st.session_state.aoi_color_mapping.items():
                    hist, x = np.histogram(df_aoi.query(f'area_of_interest == "{area}"')['abs(t)'], bins=100)
                    sign_ratio = 1 - np.cumsum(hist) / np.sum(hist)
                    fig.add_trace(go.Scattergl(x=(x[:-1] + x[1:])/2, 
                                                y=sign_ratio,
                                                mode='lines',
                                                line=dict(color=color),
                                                name=area,
                                                legendgroup=area,
                                                showlegend=col==0 and row ==0,
                                                hovertemplate=
                                                    '%s, n = %s<br>' % (area, np.sum(hist)) +
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
    fig.update_layout(width=min(2000, 2000/6*len(periods)), height=200 + 200 * len(paras),
                    font_size=17, hovermode='closest',
                    )
    fig.update_annotations(font_size=20)
        
    return fig


# --- Model comparison ---

chosen_id = stx.tab_bar(data=[
                            stx.TabBarItemData(id="tab1", title="1. Model comparison", description=""),
                            stx.TabBarItemData(id="tab2", title="2. t-distribution, compare models", description=""),
                            stx.TabBarItemData(id="tab3", title="3. t-distribution, compare areas", description=""),
                            ], 
                        default="tab2")

if chosen_id == 'tab1':
    st.markdown('#### Model comparison, all units')
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
    st.markdown('#### Distribution of t-values, compare models')
    paras = ['relative_action_value_ic', 'total_action_value', 'rpe',
            'choice_ic', 'choice_ic_next', 'trial_normalized', 'firing_1_back']

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
        df_aoi = st.session_state.df['df_ephys_units'].set_index(unit_key_names)
        df_period_linear_fit = df_period_linear_fit_all.loc[df_aoi.query('area_of_interest in @aois').index, :]
        st.markdown(f'#### N = {len(df_period_linear_fit)}')
        fig = plot_t_distribution(df_period_linear_fit=df_period_linear_fit, 
                                  periods=[p for p in all_periods if period_mapping[p] in periods], 
                                  paras=[p for p in all_paras if para_mapping[p] in paras],
                                  to_compare='models',
                                  )

        plotly_events(fig, override_height=fig.layout.height*1.1, override_width=fig.layout.width, click_event=False)

elif chosen_id == 'tab3':
    
    # --- t-distribution, compare areas ---
    st.markdown('#### Distribution of t-values, compare areas')
    paras = ['relative_action_value_ic', 'total_action_value', 'rpe',
            'choice_ic', 'choice_ic_next', 'trial_normalized', 'firing_1_back']

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
    