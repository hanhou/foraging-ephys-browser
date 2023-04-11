import streamlit as st
ss = st.session_state
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from util.streamlit_util import (filter_dataframe, aggrid_interactive_table_units, add_unit_selector, 
                            add_unit_filter, unit_plot_settings, draw_selected_units)
from streamlit_plotly_events import plotly_events

import importlib
ccf_view = importlib.import_module('.2_CCF_view', package='pages')
uplf = importlib.import_module('.1_Linear_model_comparison', package='pages')


import s3fs
from PIL import Image, ImageColor

from Home import init, select_t_sign_level

cache_fig_drift_metrics_folder = 'aind-behavior-data/Han/ephys/report/unit_drift_metrics/'
cache_fig_psth_folder = 'aind-behavior-data/Han/ephys/report/all_units/'
fs = s3fs.S3FileSystem(anon=False)

if 'df' not in ss: 
    init()

user_color_mapping = px.colors.qualitative.Plotly  # If not ccf color, use this color mapping

@st.cache_data(ttl=24*3600, show_spinner=False)
def plot_scatter(data, size=10, opacity=0.5, equal_axis=False, show_diag=False, if_ccf_color=True, **kwarg):
    df_xy = xy_to_plot['x']['column_selected'].join(xy_to_plot['y']['column_selected'], rsuffix='_y')
    df_xy.columns = ['x', 'y']
    
    x_name, y_name = data['x']['column_to_map_name'], data['y']['column_to_map_name']
    
    fig = go.Figure()
    
    if 't_' in x_name:
        fig.add_vline(x=ss['t_sign_level'], line_width=1, line_dash="dash", line_color="black")
        if not data['x']['if_take_abs']: 
            fig.add_vline(x=-ss['t_sign_level'], line_width=1, line_dash="dash", line_color="black")
    if 't_' in y_name:
        fig.add_hline(y=ss['t_sign_level'], line_width=1, line_dash="dash", line_color="black")
        if not data['y']['if_take_abs']:
            fig.add_hline(y=-ss['t_sign_level'], line_width=1, line_dash="dash", line_color="black")
    
    if show_diag:
        _min = df_xy.values.ravel().min()
        _max = df_xy.values.ravel().max()
        fig.add_trace(go.Scattergl(x=[_min, _max], 
                                   y=[_min, _max], mode='lines',
                                   line=dict(dash='dash', color='gray', width=1.1),
                                   showlegend=False)
                      )
    
    if equal_axis:
        fig.update_yaxes(
            scaleanchor = "x",
            scaleratio = 1,
        )   
    
    if if_ccf_color:
        df_color = pd.DataFrame.from_dict(ss.aoi_color_mapping, orient='index', columns=['colors'])
    else:
        df_color = pd.DataFrame.from_dict({aoi: user_color_mapping[i%len(user_color_mapping)] for i, aoi in enumerate(ss.aoi_color_mapping)}, orient='index', columns=['colors'])
    
    # Batch define color
    df_xy = df_xy.join(df_color, on='area_of_interest')  # all use normal colors
    df_xy['opacity'] = opacity        

    if len(ss.df_selected_from_xy_view):  # If there are selected dots, put unselcted dots to gray
        un_selected = ~df_xy.reset_index().set_index(ss.unit_key_names).index.isin(ss.df_selected_from_xy_view.reset_index().set_index(ss.unit_key_names).index)
        df_xy.loc[un_selected, 'colors'] = 'lightgrey'
        df_xy.loc[un_selected, 'opacity'] = 0.3
    
    # For each aoi, plot the dots
    for i, aoi in enumerate([aoi for aoi in ss.df['aoi'].index 
                             if aoi in df_xy.reset_index().area_of_interest.values]):
        
        df_xy_this = df_xy.query(f'area_of_interest == "{aoi}"').reset_index()
        df_xy_this = pd.concat([df_xy_this.query('colors != "lightgrey"'), df_xy_this.query('colors == "lightgrey"')])
    
        fig.add_trace(go.Scattergl(x=df_xy_this.x, 
                                y=df_xy_this.y,
                                mode="markers",
                                marker=dict(symbol='circle', size=size, opacity=df_xy_this.opacity.values, 
                                                line=dict(color='white', width=1),
                                                color=df_xy_this.colors.values), 
                                name=aoi,
                                hovertemplate=  '%s' % aoi +
                                                ' (uid = %{customdata[0]})<br>' +
                                                '%{customdata[1]}, s%{customdata[2]}, i%{customdata[3]}, u%{customdata[4]}<br>' +
                                                '%s = %%{x:.4g}<br>%s = %%{y:.4g}<extra></extra>' % (x_name, y_name),
                                customdata=np.stack((df_xy_this.uid, df_xy_this.h2o, 
                                                        df_xy_this.session, df_xy_this.insertion_number, df_xy_this.unit), axis=-1),
                                unselected=dict(marker_color='lightgrey'),
                                )
                    )
       
        
    x_period = f", {uplf.period_name_mapper[data['x']['column_to_map'][0]]}" if isinstance(data['x']['column_to_map'], list) else ''
    y_period = f", {uplf.period_name_mapper[data['y']['column_to_map'][0]]}" if isinstance(data['y']['column_to_map'], list) else ''
    
    fig.update_layout(width=1000, height=900, font=dict(size=20), 
                      hovermode='closest', showlegend=True, dragmode='select',
                      xaxis_title=x_name + x_period, 
                      yaxis_title=y_name + y_period,)
            
    return fig



def add_xy_selector():
    
    with st.expander('XY selector', expanded=True):
        col3, col4 = st.columns([1, 1])
        with col3:
            st.markdown('### X axis')
            xy_selected = {'x': ccf_view.select_para_of_interest(prompt='Type', suffix='_x',
                                                                 default_model='dQ, sumQ, rpe, C*2, R*5, t',
                                                                 default_period='iti_all',
                                                                 default_paras='relative_action_value_ic',)}
        with col4:
            st.markdown('### Y axis')
            xy_selected.update(y=ccf_view.select_para_of_interest(prompt='Type', suffix='_y',
                                                                 default_model='dQ, sumQ, rpe, C*2, R*5, t',
                                                                 default_period='iti_all',
                                                                 default_paras='total_action_value',))
            
        # st.form_submit_button("update axes")

    return xy_selected



# --------------------------------
if __name__ == '__main__':
    with st.sidebar:
        try:
            add_unit_filter()
        except:
            st.experimental_rerun()
        
        with st.expander('t-value threshold', expanded=True):
            select_t_sign_level()
        
        add_unit_selector()

    ss.aggrid_outputs = aggrid_interactive_table_units(df=ss.df_unit_filtered, height=300)
    col1, _, col2 = st.columns((1, 0.1, 1.5))

    with col1:
        xy_to_plot = add_xy_selector()
        
        with st.expander('plot settings', expanded=True):
            cols = st.columns([1, 1, 0.7])
            size = cols[0].slider('dot size', 1, 30, step=1, value=10)
            opacity = cols[1].slider('opacity', 0.0, 1.0, step=0.05, value=0.7)
            if_ccf_color = cols[2].checkbox('use ccf color', value=True)
            equal_axis = cols[2].checkbox('equal axis', value=xy_to_plot['x']['column_to_map'][2] == xy_to_plot['y']['column_to_map'][2])
            show_diag = cols[2].checkbox('show diagonal', value=xy_to_plot['x']['column_to_map_name'] == xy_to_plot['y']['column_to_map_name'])
            
            
        # for i in range(2): st.write('\n')

        st.markdown("---")
        if_draw_units = unit_plot_settings(need_click=True)
        
    if len(xy_to_plot['x']['column_selected']):
        with col2:
            fig = plot_scatter(xy_to_plot, size=size, opacity=opacity, 
                               equal_axis=equal_axis, show_diag=show_diag, if_ccf_color=if_ccf_color,
                               state=ss.df_selected_from_xy_view   # Trigger replot when df_selected_from_xy_view changes, otherwise, the plot is cached
                               )

            # Select other Plotly events by specifying kwargs
            selected_points_xy_view = plotly_events(fig, click_event=True, hover_event=False, select_event=True,
                                                    override_height=fig.layout.height*1.1, 
                                                    override_width=fig.layout.width, key='unit_scatter')

        if len(selected_points_xy_view):
            df_xy = xy_to_plot['x']['column_selected'].join(xy_to_plot['y']['column_selected'])
            df_xy.columns = ['x', 'y']
            df_selected_from_xy_view = pd.concat([df_xy.query(f'x == {xy["x"]} and y == {xy["y"]}') 
                                                            for xy in selected_points_xy_view], axis=0)
            
            # If selected units change, rerun the whole app
            if len(df_selected_from_xy_view) and not (set(df_selected_from_xy_view.index) == 
                                                      set(ss.df_selected_from_xy_view.index)):
                ss.df_selected_from_xy_view = df_selected_from_xy_view
                st.experimental_rerun()
        
            if if_draw_units:
                draw_selected_units()