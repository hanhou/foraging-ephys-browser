import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm

import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

from streamlit_plotly_events import plotly_events

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
period_mapping = {'before_2': 'Before GO (2s)', 'delay': 'Delay', 'go_1.2': 'After GO (1.2s)', 'go_to_end': 'GO to END', 
                  'iti_all': 'ITI (all)', 'iti_first_2': 'ITI (first 2s)', 'iti_last_2': 'ITI (last 2s)'}

all_paras = [var for var in df_period_linear_fit_all.columns.get_level_values('var_name').unique() if var !='']
para_mapping = {'relative_action_value_ic': 'dQ', 'total_action_value': 'sumQ', 'rpe': 'rpe',
                'choice_ic': 'Choice (this)', 'choice_ic_next': 'Choice (next)',
                'trial_normalized': 'Trial number', 
                **{f'firing_{n}_back': f'Firing {n} back' for n in range(1, 11)},
                'contra_action_value': 'contraQ', 'ipsi_action_value': 'ipsiQ'
                }

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
def plot_t_distribution(df_period_linear_fit, periods, paras):
    models = [m for m in all_models if 'dQ' in m]

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
            df = df_period_linear_fit.loc[:, (period, models, "t", para)
                                            ].droplevel(axis=1, level=0).stack([0, 2]).reset_index()
            df['abs(t)'] = df.t.abs()

            for i, model in enumerate(models):
                hist, x = np.histogram(df.query(f'multi_linear_model == "{model}"')['abs(t)'], 
                                    bins=100)
                sign_ratio = 1 - np.cumsum(hist) / np.sum(hist)
                fig.add_trace(go.Scattergl(x=(x[:-1] + x[1:])/2, 
                                        y=sign_ratio,
                                        mode='lines',
                                        line=dict(color=model_color_map[model],
                                                width=5 if model in ('dQ, sumQ, rpe', 'dQ, sumQ, rpe, C*2, R*5, t') else 3),
                                        name=model,
                                        legendgroup=model,
                                        showlegend=col==0 and row ==0,
                                        hovertemplate=
                                            '%s<br>' % (model) +
                                            '%{y:%2.1f} units, t > %{x:.2f}<br><extra></extra>',
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
                                    ),
                        row=row+1, col=col+1)

            fig.add_vline(x=2.0, line_color='gray', line_dash='dash',
                        row=row+1, col=col+1)

    for row, para in enumerate(paras):
        fig['layout'][f'yaxis{1 + row * len(periods)}']['title'] = para_mapping[para]

    # fig.update_traces(line_width=3)
    fig.update_xaxes(range=[0, 10])
    fig.update_yaxes(range=[0, 1.1])
    fig.update_layout(width=2000, height=250 * len(paras),
                    font_size=17, hovermode='closest',
                    )
    fig.update_annotations(font_size=20)
        
    return fig


# --- Model comparison ---
st.markdown('### Model comparison, all units')
if st.checkbox('do it', False):
    fig = plot_model_comparison()

    fig.update_layout(width=2000, height=700, 
                    #   title='Model comparison for unit period fitting',
                    xaxis_title='Period',
                    yaxis_title='Model BIC',
                    **plotly_font(20)
                    )

    fig.update_xaxes(categoryorder='array', categoryarray=all_models)
    st.plotly_chart(fig)

# --- Compare (dQ, sumQ, rpe) vs (full mode: C*2, R*5, t) ---

# 1. t value of dQ and sum Q
# st.markdown('### t-values of (dQ, sumQ, rpe), from different models')
# period = st.columns([1, 6])[0].selectbox('period', all_periods, index=all_periods.index('iti_all'))
# fig = t_value_Q_rpe_all(df_period_linear_fit_all, period=period)
# plotly_events(fig, override_height=fig.layout.height*1.1, override_width=fig.layout.width, click_event=False)

# 2. t value of dQ and sum Q, different epochs
st.markdown('### t-values, all units, different epoches and models')
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

cols = st.columns([1, 1])
aois = cols[0].multiselect('Areas to include', st.session_state.aoi_color_mapping.keys(), st.session_state.aoi_color_mapping)
paras = cols[1].multiselect('Variables to draw', all_paras, ['relative_action_value_ic', 'total_action_value', 'rpe',
                                                        'choice_ic', 'choice_ic_next'])

if aois and paras:
    df_aoi = st.session_state.df['df_ephys_units'].set_index(unit_key_names)
    df_period_linear_fit = df_period_linear_fit_all.loc[df_aoi.query('area_of_interest in @aois').index, :]
    st.markdown(f'#### N = {len(df_period_linear_fit)}')
    fig = plot_t_distribution(df_period_linear_fit=df_period_linear_fit, 
                             periods=all_periods, 
                             paras=paras)

    plotly_events(fig, override_height=fig.layout.height*1.1, override_width=fig.layout.width, click_event=False)
    
    