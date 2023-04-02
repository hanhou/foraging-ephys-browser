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
from scipy.stats import norm

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
 

def add_unit_filter():
    with st.expander("Unit filter", expanded=True):   
        st.session_state.df_unit_filtered = filter_dataframe(df=st.session_state.df['df_ephys_units'])
        # Join with df_period_linear_fit_all here! (A huge dataframe with all things merged (flattened multi-level columns)
        st.session_state.df_unit_filtered_merged = st.session_state.df_unit_filtered.set_index(st.session_state.unit_key_names
                                                                                        ).join(st.session_state.df['df_period_linear_fit_all'], how='inner')
    
    n_units = len(st.session_state.df_unit_filtered)
    n_animal = len(st.session_state.df_unit_filtered['subject_id'].unique())
    n_insertion = len(st.session_state.df_unit_filtered.groupby(['subject_id', 'session', 'insertion_number']))
                      
    st.markdown(f"### Filtered: {len(st.session_state.df_unit_filtered)} units, {n_animal} animals, {n_insertion} insertions")


# For pure units
pure_unit_color_mapping =  {'pure_dQ': 'darkviolet',
                            'pure_sumQ': 'deepskyblue',
                            'pure_contraQ': 'darkblue',
                            'pure_ipsiQ': 'darkorange'}
                                
polar_classifiers = {'dQ, sumQ, rpe': [{'x_name': 'relative_action_value_ic', 'y_name': 'total_action_value'},
                                        {'pure_dQ': [(-22.5, 22.5), (-22.5 + 180, 180), (-180, -180 + 22.5)],
                                        'pure_sumQ': [(22.5 + 45, 67.5 + 45), (22.5 + 45 - 180, 67.5 + 45 - 180)],
                                        'pure_contraQ': [(22.5, 67.5), (22.5 - 180, 67.5 - 180)],
                                        'pure_ipsiQ': [(22.5 + 90, 67.5 + 90), (22.5 + 90 - 180, 67.5 + 90 - 180)]}],
                        
                     'contraQ, ipsiQ, rpe':  [{'x_name': 'ipsi_action_value', 'y_name': 'contra_action_value'},
                                            {'pure_dQ': [(22.5 + 90, 67.5 + 90), (22.5 + 90 - 180, 67.5 + 90 - 180)],
                                            'pure_sumQ': [(22.5, 67.5), (22.5 - 180, 67.5 - 180)],
                                            'pure_contraQ': [(22.5 + 45, 67.5 + 45), (22.5 + 45 - 180, 67.5 + 45 - 180)],
                                            'pure_ipsiQ': [(-22.5, 22.5), (-22.5 + 180, 180), (-180, -180 + 22.5)],}]
}

def _to_theta_r(x, y):
    return np.rad2deg(np.arctan2(y, x)), np.sqrt(x**2 + y**2)

def compute_pure_polar_classification(t_sign_level=1.96):
    # Add or update pure polar classification in st.session_state.df['df_period_linear_fit_all']
    df = st.session_state.df['df_period_linear_fit_all']  # assign pointer
    
    for period in df.columns.get_level_values('period').unique():
        for model in df.columns.get_level_values('multi_linear_model').unique():
            classfier_key = [k for k in polar_classifiers if k in model][0]
            
            # Compute the proportions of pure neurons from dQ / sumQ 
            x = df.loc[:, (period, model, 't', polar_classifiers[classfier_key][0]['x_name'])].values
            y = df.loc[:, (period, model, 't', polar_classifiers[classfier_key][0]['y_name'])].values
            theta, _ = _to_theta_r(x, y)

            for unit_class, ranges in polar_classifiers[classfier_key][1].items():
                if_pure_this = np.any([(a_min < theta) & (theta < a_max) for a_min, a_max in ranges], axis=0) 
                if_pure_this = if_pure_this & (np.sqrt(x ** 2 + y ** 2) >= t_sign_level)
                df[period, model, f'{unit_class}', ''] = if_pure_this.astype(int)

# --- t and p-value threshold ---

t_to_p = lambda t: 2 * norm.sf(t)
p_to_t = lambda p: norm.ppf(1 - p / 2)

def _on_change_t_sign_level():
    compute_pure_polar_classification(t_sign_level=st.session_state['t_sign_level'])
    
def select_t_sign_level(col=st):
    t_now = st.session_state['t_sign_level'] if 't_sign_level' in st.session_state else 1.96   # Note: use ['xxx'] instead of .xxx!!
    return col.slider(f'Significant level = t > {t_now:.3g} (p < {t_to_p(t_now):.2g})', 
                      1.0, 5.0, 
                      value=t_now,
                      key='t_sign_level',
                      on_change=_on_change_t_sign_level)
    # return col.number_input(f"significant level: t >= {st.session_state['t_sign_level']}", 
    #                                                  value=st.session_state['t_sign_level'], 
    #                                                  min_value=0.0, step=0.1,
    #                                                 key='t_sign_level',
    #                                                 on_change=_on_change_t_sign_level)   # Update pure polar classification

def init():
    st.set_page_config(layout="wide", page_title='Foraging ephys browser',
                       page_icon='⚡',
                       menu_items={
                        'Report a bug': "https://github.com/hanhou/foraging-ephys-browser/issues",
                        'About': "Github repo: https://github.com/hanhou/foraging-ephys-browser/",
                    })

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
    
    # Add pure polar classification
    compute_pure_polar_classification()
    
    # Other stuff
    st.session_state['selected_points'] = []
    
        
# ------- Layout starts here -------- #    

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