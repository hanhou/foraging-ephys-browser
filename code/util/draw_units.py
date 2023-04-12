import streamlit as st
ss = st.session_state

from datetime import datetime 

import s3fs
from PIL import Image, ImageColor

cache_fig_drift_metrics_folder = 'aind-behavior-data/Han/ephys/report/unit_drift_metrics/'
cache_fig_psth_folder = 'aind-behavior-data/Han/ephys/report/all_units/'
fs = s3fs.S3FileSystem(anon=False)



def unit_draw_settings(default_source='xy_view', need_click=True):
    
    cols = st.columns([3, 1])
    
    ss.unit_select_source = cols[0].selectbox('Which unit(s) to draw?', 
                                            [f'selected from {source} '
                                             f'({len(ss[f"df_selected_from_{source}"])} units)' 
                                             for source in ss.select_sources], 
                                            index=ss.select_sources.index(default_source)
                                            )
        
    # cols[0].markdown(f'##### Show selected {len(df_selected)} unit(s)')
    ss.num_cols = cols[1].number_input('Number of columns', 1, 10, 
                                                     ss.num_cols if 'num_cols' in ss else 3)

    ss.draw_types = st.multiselect('Which plot(s) to draw?', ['psth', 'drift metrics'], 
                                                 default=ss.draw_types if 'draw_types' in ss else ['psth'])
    
    if need_click:
        cols = st.columns([1, 3])
        ss.auto_draw = cols[0].checkbox('Auto draw', value=ss.auto_draw if 'auto_draw' in ss else False)
        draw_it = cols[1].button(f'================ 🎨 Draw! ================', use_container_width=True)
    else:
        draw_it = True
    return draw_it or ss.auto_draw



# @st.cache_data(max_entries=100)
def get_fig_unit_psth_only(key):
    fn = f'*{key["h2o"]}_{key["session_date"]}_{key["insertion_number"]}*u{key["unit"]:03}*'
    aoi = key["area_of_interest"]
    
    file = fs.glob(cache_fig_psth_folder + ('' if aoi == 'others' else aoi + '/') + fn)
    if len(file) == 1:
        with fs.open(file[0]) as f:
            img = Image.open(f)
            img = img.crop((500, 140, 3000, 2800)) 
    else:
        img = None
            
    return img

# @st.cache_data(max_entries=100)
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


def draw_selected_units():    
    
    for source in ss.select_sources:
        if source in ss.unit_select_source: break
        
    # Add ccf_z
    df_selected = ss[f'df_selected_from_{source}'].reset_index()
    if 'ccf_z' not in df_selected.columns:
        df_selected = df_selected.merge(ss.df_unit_filtered[['uid', 'ccf_z']], on='uid')
    
    st.write(f'Loading selected {len(df_selected)} units...')
    my_bar = st.columns((1, 7))[0].progress(0)

    cols = st.columns([1]*ss.num_cols)
    
    sort_by_col = [col for col in df_selected.columns if 'sort_by' in col]
    
    for i, key in enumerate(df_selected.to_dict(orient='records')):
        key['session_date'] = datetime.strftime(datetime.strptime(str(key['session_date']), '%Y-%m-%d %H:%M:%S'), '%Y%m%d')
        col = cols[i%ss.num_cols]
        
        sort_str = f'''<br>[{sort_by_col[0]} = {key[sort_by_col[0]]:.2f}]''' if sort_by_col else ''''''
        col.markdown(f'''<h5 style='text-align: center; color: orange;'>'''
                     f'''{"Left" if key["ccf_z"] < 5700 else "Right"} {key["area_of_interest"]}'''
                     f''' ({key["h2o"]}s{key["session"]}i{key['insertion_number']}u{key["unit"]}, uid={key['uid']})'''
                     f'''{sort_str}''',
            unsafe_allow_html=True)

        for draw_type in ss.draw_types:
            img = draw_func_mapping[draw_type](key)
            if img is None:
                col.markdown(f'{draw_type} fetch error')
            else:
                col.image(img, output_format='PNG', use_column_width=True)
        
        col.markdown("---")
        my_bar.progress(int((i + 1) / len(df_selected) * 100))