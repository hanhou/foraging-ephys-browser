import streamlit as st

import plotly.graph_objs as go
import plotly.express as px


unit_key_names = ['subject_id', 'session', 'insertion_number', 'unit']

model_order = ['dQ, sumQ, rpe', 
               'dQ, sumQ, rpe, C*2', 
               'dQ, sumQ, rpe, C*2, t', 
               'dQ, sumQ, rpe, C*2, R*1', 
               'dQ, sumQ, rpe, C*2, R*1, t',
               'dQ, sumQ, rpe, C*2, R*5, t',
               'dQ, sumQ, rpe, C*2, R*10, t',
               'contraQ, ipsiQ, rpe',
               'contraQ, ipsiQ, rpe, C*2, R*5, t']

periods = ['before_2', 'iti_all', 'go_to_end']


@st.cache_data(ttl=3600*24)
def plot_model_comparison():
    df_period_linear_fit_all = st.session_state.df['df_period_linear_fit_all']
    df_period_linear_fit_melt = df_period_linear_fit_all.iloc[:, df_period_linear_fit_all.columns.get_level_values(-2)=='rel_bic'
                                                            ].stack(level=[0, 1]
                                                                    ).droplevel(axis=1, level=1
                                                                            ).reset_index()
    fig = px.box(df_period_linear_fit_melt.query('multi_linear_model in @model_order'), x='period', y='rel_bic', 
                color='multi_linear_model', category_orders={"multi_linear_model": model_order})
    return fig


fig = plot_model_comparison()
fig.update_layout(width=2000, height=700, font=dict(size=20))
fig.update_xaxes(categoryorder='array', categoryarray=model_order)
    
st.plotly_chart(fig)