#%%
import pandas as pd
import streamlit as st
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import s3fs
import os

from PIL import Image, ImageColor
import streamlit.components.v1 as components
import streamlit_nested_layout

from streamlit_util import filter_dataframe, aggrid_interactive_table_units


if_profile = False

if if_profile:
    from streamlit_profiler import Profiler
    p = Profiler()
    p.start()


# from pipeline import experiment, ephys, lab, psth_foraging, report, foraging_analysis
# from pipeline.plot import foraging_model_plot

cache_folder = 'xxx' # '/Users/han.hou/s3-drive/st_cache/'
cache_fig_folder = 'xxx' # '/Users/han.hou/Library/CloudStorage/OneDrive-AllenInstitute/pipeline_report/report/all_units/'  # 

if os.path.exists(cache_folder):
    use_s3 = False
else:
    cache_folder = 'aind-behavior-data/Han/ephys/report/st_cache/'
    cache_fig_folder = 'aind-behavior-data/Han/ephys/report/all_units/'
    
    fs = s3fs.S3FileSystem(anon=False)
    use_s3 = True

if 'selected_points' not in st.session_state:
    st.session_state['selected_points'] = []

    
@st.cache_data(ttl=24*3600)
def load_data(tables=['sessions']):
    df = {}
    for table in tables:
        file_name = cache_folder + f'{table}.pkl'
        if use_s3:
            with fs.open(file_name) as f:
                df[table] = pd.read_pickle(f)
        else:
            df[table] = pd.read_pickle(file_name)
        
    return df

# @st.cache_data(ttl=24*3600)
def get_fig_unit_all_in_one(key):
    sess_date_str = datetime.strftime(datetime.strptime(key['session_date'], '%Y-%m-%dT%H:%M:%S'), '%Y%m%d')
    
    fn = f'*{key["h2o"]}_{sess_date_str}_{key["ins"]}*u{key["unit"]:03}*'
    aoi = key["area_of_interest"]
    
    if use_s3:
        file = fs.glob(cache_fig_folder + ('' if aoi == 'others' else aoi + '/') + fn)
        if len(file) == 1:
            with fs.open(file[0]) as f:
                img = Image.open(f)
                img = img.crop((500, 140, 5400, 3000))           
    else:
        file = glob.glob(cache_fig_folder + ('' if aoi == 'others' else aoi + '/') + fn)
        if len(file) == 1:
            img = Image.open(file[0])
            img = img.crop((500, 140, 5400, 3000))
            
    return img

# table_mapping = {
#     'sessions': fetch_sessions,
#     'df_ephys_units': fetch_df_ephys_units,
# }


@st.cache_data(ttl=24*3600)
def get_fig(key):
    fig = plt.figure(figsize=(8, 3), constrained_layout=True)
    ax = fig.subplots(1,1)
    
    foraging_model_plot.plot_session_fitted_choice(key, ax=ax, remove_ignored=False, first_n=2)
    return fig   

def add_unit_filter():
    with st.expander("Unit filter", expanded=True):   
        st.session_state.df_unit_filtered = filter_dataframe(df=st.session_state.df['df_ephys_units'])
        # Join with df_period_linear_fit_all here! (A huge dataframe with all things merged (flattened multi-level columns)
        st.session_state.df_unit_filtered_merged = st.session_state.df_unit_filtered.set_index(st.session_state.unit_key_names
                                                                                        ).join(st.session_state.df['df_period_linear_fit_all'], how='inner')
        
    st.markdown(f"### {len(st.session_state.df_unit_filtered)} units filtered")


# ------- Layout starts here -------- #    
def init():
    st.set_page_config(layout="wide", page_title='Foraging unit navigator')

    df = load_data(['sessions', 'df_ephys_units', 'aoi', 'df_period_linear_fit_all'])
    st.session_state.df = df
    st.session_state.unit_key_names = ['subject_id', 'session', 'insertion_number', 'unit']

    st.session_state.aoi_color_mapping = {area: f'rgb({",".join(col.astype(str))})' for area, col in zip(df['aoi'].index, df['aoi'].rgb)}
    # Some global variables
    st.session_state.scatter_stats_names = [keys for keys in st.session_state.df['df_ephys_units'].keys() if any([s in keys for s in 
                                                                                                                ['dQ', 'sumQ', 'contraQ', 'ipsiQ', 'rpe', 'ccf', 'firing_rate',
                                                                                                                'poisson']])]
    
    st.session_state.df['df_ephys_units']['number_units'] = 1
    unit_qc = ['number_units', 'unit_amp', 'unit_snr', 'presence_ratio', 'amplitude_cutoff', 'isi_violation', 'avg_firing_rate',
               'poisson_p_choice_outcome', 'poisson_p_dave']
    st.session_state.ccf_stat_names = unit_qc
        
def app():
    st.markdown('## Foraging Unit Browser')
       
          
    with st.container():
        # col1, col2 = st.columns([1.5, 1], gap='small')
        # with col1:
        # -- 1. unit dataframe --
        st.markdown('### Filtered units')
        
        # aggrid_outputs = aggrid_interactive_table_units(df=df['df_ephys_units'])
        # st.session_state.df_unit_filtered = aggrid_outputs['data']
        
        container_filtered_frame = st.container()
        
    with st.sidebar:
        add_unit_filter()
        
    st.session_state.aggrid_outputs = aggrid_interactive_table_units(df=st.session_state.df_unit_filtered)

    # st.dataframe(st.session_state.df_unit_filtered, use_container_width=True, height=1000)

    container_unit_all_in_one = st.container()
    
    with container_unit_all_in_one:
        # with st.expander("Expand to see all-in-one plot for selected unit", expanded=True):
        if len(st.session_state.aggrid_outputs['selected_rows']) == 1:
            unit_fig = get_fig_unit_all_in_one(st.session_state.aggrid_outputs['selected_rows'][0])
            st.image(unit_fig, output_format='PNG', width=3000)

if 'df' not in st.session_state: 
    init()
    
if __name__ == '__main__':
    app()

            
if if_profile:    
    p.stop()