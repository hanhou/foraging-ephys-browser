import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from streamlit_util import filter_dataframe, aggrid_interactive_table_units
from streamlit_plotly_events import plotly_events

from datetime import datetime 

import importlib
ccf_view = importlib.import_module('.2_CCF_view', package='pages')

import s3fs
from PIL import Image, ImageColor

from Home import add_unit_filter, init, select_t_sign_level


cache_fig_drift_metrics_folder = 'aind-behavior-data/Han/ephys/report/unit_drift_metrics/'
cache_fig_psth_folder = 'aind-behavior-data/Han/ephys/report/all_units/'
fs = s3fs.S3FileSystem(anon=False)

if 'df' not in st.session_state: 
    init()

@st.cache_data(ttl=24*3600)
def plot_scatter(data, x_name='dQ_iti', y_name='sumQ_iti', if_use_ccf_color=False, sign_level=1.96, x_abs=False, y_abs=False):
    
    fig = go.Figure()
    
    for aoi in st.session_state.df['aoi'].index:
        if aoi not in st.session_state.df_unit_filtered.area_of_interest.values:
            continue
        
        this_aoi = data.query(f'area_of_interest == "{aoi}"')
        fig.add_trace(go.Scattergl(x=np.abs(this_aoi[x_name]) if x_abs else this_aoi[x_name], 
                                 y=np.abs(this_aoi[y_name]) if y_abs else this_aoi[y_name],
                                 mode="markers",
                                 marker_color=st.session_state.aoi_color_mapping[aoi] if if_use_ccf_color else None,
                                 name=aoi))
        
    # fig = px.scatter(data, x=x_name, y=y_name, 
    #                 color='area_of_interest', symbol="area_of_interest",
    #                 hover_data=['annotation'],
    #                 color_discrete_map=aoi_color_mapping if if_use_ccf_color else None)
    
    if 't_' in x_name:
        fig.add_vline(x=sign_level, line_width=1, line_dash="dash", line_color="black")
        if not x_abs: 
            fig.add_vline(x=-sign_level, line_width=1, line_dash="dash", line_color="black")
    if 't_' in y_name:
        fig.add_hline(y=sign_level, line_width=1, line_dash="dash", line_color="black")
        if not y_abs:
            fig.add_hline(y=-sign_level, line_width=1, line_dash="dash", line_color="black")
        
    # fig.update_xaxes(range=[-40, 40])
    # fig.update_yaxes(range=[-40, 40])
    
    fig.update_layout(width=900, height=800, font=dict(size=20), xaxis_title=x_name, yaxis_title=y_name)
    
    if all(any([s in name for s in ['t_', 'beta_']]) for name in [x_name, y_name]):
        fig.update_yaxes(
            scaleanchor = "x",
            scaleratio = 1,
        )
            
    return fig

def add_scatter_return_selected(data, x_name, y_name, x_abs=False, y_abs=False):
    if len(data):
        if_use_ccf_color = True # st.checkbox("Use ccf color", value=True)
        fig = plot_scatter(data, x_name=x_name, y_name=y_name, 
                            if_use_ccf_color=if_use_ccf_color, 
                            sign_level=st.session_state['t_sign_level'],
                            x_abs=x_abs,
                            y_abs=y_abs)
        
        if len(st.session_state.selected_points):
            fig.add_trace(go.Scattergl(x=[st.session_state.selected_points[0]['x']], 
                                     y=[st.session_state.selected_points[0]['y']], 
                                mode='markers',
                                marker_symbol='star',
                                marker_size=15,
                                marker_color='black',
                                name='selected'))
            
        # if 'selected_points_scatter' in st.session_state and len(st.session_state.selected_points_scatter):
        #     fig.add_trace(go.Scatter(x=[pt['x'] for pt in st.session_state.selected_points_scatter], 
        #                         y=[pt['y'] for pt in st.session_state.selected_points_scatter], 
        #                         mode='markers',
        #                         marker_symbol='star',
        #                         marker_size=15,
        #                         marker_color='black',
        #                         name='selected'))    
        
        fig.update_layout(height=900, width=1000, font=dict(size=20),
                          hovermode='closest',)       
        
        # Select other Plotly events by specifying kwargs
        selected_points_scatter = plotly_events(fig, click_event=True, hover_event=False, select_event=True,
                                                override_height=fig.layout.height*1.1, 
                                                override_width=fig.layout.width, key='unit_scatter')
    return selected_points_scatter


def get_fig_unit_psth_only(key):
    sess_date_str = datetime.strftime(datetime.strptime(key['session_date'], '%Y-%m-%dT%H:%M:%S'), '%Y%m%d')
    
    fn = f'*{key["h2o"]}_{sess_date_str}_{key["insertion_number"]}*u{key["unit"]:03}*'
    aoi = key["area_of_interest"]
    
    file = fs.glob(cache_fig_psth_folder + ('' if aoi == 'others' else aoi + '/') + fn)
    if len(file) == 1:
        with fs.open(file[0]) as f:
            img = Image.open(f)
            img = img.crop((500, 140, 3000, 2800)) 
    else:
        img = None
            
    return img


def get_fig_unit_drift_metric(key):
    fn = f'*{key["subject_id"]}_{key["session"]}_{key["insertion_number"]}_{key["unit"]:03}*'
    
    file = fs.glob(cache_fig_drift_metrics_folder + fn)
    if len(file) == 1:
        with fs.open(file[0]) as f:
            img = Image.open(f)
            img = img.crop((0, 0, img.size[0], img.size[1]))  
    else:
        img = None
            
    return img

draw_func_mapping = {'psth': get_fig_unit_psth_only,
                     'drift metrics': get_fig_unit_drift_metric}

def draw_selected_units(selected_points, draw_types, x_name, y_name, x_abs, y_abs):
    st.write(f'Draw unit plot for selected {len(selected_points)} units')
    my_bar = st.columns((1, 7))[0].progress(0)

    cols = st.columns((1, 1, 1))
    
    for i, xy in enumerate(selected_points):
        q_x = f'abs({x_name}) == {xy["x"]}' if x_abs else f'{x_name} == {xy["x"]}'
        q_y = f'abs({y_name}) == {xy["y"]}' if y_abs else f'{y_name} == {xy["y"]}'
        key = st.session_state.df_unit_filtered.query(f'{q_x} and {q_y}')
        if len(key):
            for draw_type in draw_types:
                img = draw_func_mapping[draw_type](key.iloc[0])
                if img is None:
                    cols[i % 3].markdown(f'{draw_type} fetch error')
                else:
                    cols[i % 3].image(img, output_format='PNG', use_column_width=True)
        else:
            cols[i % 3].markdown('Unit not found')
            
        cols[i % 3].markdown("---")
        my_bar.progress(int((i + 1) / len(selected_points) * 100))
    pass


def add_xy_selector():
    
    with st.form(key='X-Y_selector'):
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
            
        st.form_submit_button("update axes")

    return xy_selected


# --------------------------------
if __name__ == '__main__':
    with st.sidebar:
        add_unit_filter()
        with st.expander('t-value threshold', expanded=True):
            select_t_sign_level()

    st.session_state.aggrid_outputs = aggrid_interactive_table_units(df=st.session_state.df_unit_filtered, height=300)
    col1, col2 = st.columns((1, 1.5))

    with col1:
        xy_selected = add_xy_selector()
        draw_types = st.multiselect('Which plot(s) to draw?', ['psth', 'drift metrics'], default=['psth'])
    with col2:
        selected_points_scatter = add_scatter_return_selected(st.session_state.df_unit_filtered, draw_types)


    draw_selected_units(selected_points_scatter, draw_types, draw_types)