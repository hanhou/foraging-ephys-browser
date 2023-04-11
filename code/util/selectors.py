import streamlit as st
ss = st.session_state

import numpy as np

from util import *
from util.streamlit_util import filter_dataframe


def select_period(multi=True, label='Periods to plot', 
                  default_period='iti_all',
                  col=st, suffix=''):
    if multi:
        period_names = col.multiselect(label, 
                                   period_name_mapper.values(),
                                   [period_name_mapper[p] for p in period_name_mapper if p!= 'delay'],
                                   key=f'period_names{suffix}')
        return period_names, [p for p in period_name_mapper if period_name_mapper[p] in period_names]
    else:
        period_name = col.selectbox(label, 
                                    period_name_mapper.values(),
                                    list(period_name_mapper.keys()).index(default_period),
                                    key=f'period_name{suffix}')
        return period_name, [p for p in period_name_mapper if period_name_mapper[p] == period_name][0]
    

def select_model(available_models=list(model_name_mapper.keys()),
                 default_model='dQ, sumQ, rpe, C*2, R*5, t',
                label='Model to plot', 
                suffix='',
                col=st):
    model_name = col.selectbox(label,
                               [model_name_mapper[m] for m in available_models], 
                                available_models.index(default_model),
                                key=f'model_name{suffix}')
    model = [m for m in model_name_mapper if model_name_mapper[m] == model_name][0]
    return model_name, model

def select_para(multi=True,
                available_paras=list(para_name_mapper.keys()), 
                default_paras='relative_action_value_ic', 
                label='Variables to plot',
                suffix='',
                col=st):
    if multi:
        para_names = col.multiselect(label, 
                                 [para_name_mapper[p] for p in available_paras],
                                 [para_name_mapper[p] for p in default_paras],
                                 key=f'para_names{suffix}'
                                 )
        return para_names, [p for p in para_name_mapper if para_name_mapper[p] in para_names]
    else:
        para_name = col.selectbox(label, 
                                  [para_name_mapper[p] for p in available_paras],
                                  available_paras.index(default_paras),
                                  key=f'para_name{suffix}'
                                  )
        return para_name, [p for p in para_name_mapper if para_name_mapper[p] == para_name][0]



def select_para_of_interest(prompt="Map what to CCF?", suffix='',
                            default_model='dQ, sumQ, rpe, C*2, R*5, t',
                            default_period='iti_all',
                            default_paras=None,):
    
    type_to_map = st.selectbox(prompt,
                            ['unit tuning', 'unit stats'],
                            key=f'type_to_map{suffix}',
                            index=0)
    
    if type_to_map == 'unit tuning':
        
        _, model = select_model(label='which model',
                                     suffix=suffix,
                                     default_model=default_model,
                                    )
        _, period = select_period(multi=False,
                                       suffix=suffix,
                                       default_period=default_period,
                                       label='which period')
        
        df_this_model = ss.df['df_period_linear_fit_all'].iloc[:, ss.df['df_period_linear_fit_all'].columns.get_level_values('multi_linear_model') == model]

        cols= st.columns([1, 1])
        stat = cols[0].selectbox("which statistic",
                                ['t', 'beta', 'model_r2', 'model_bic'] + list(pure_unit_color_mapping.keys()), 
                                0,
                                key=f'stat{suffix}')  # Could be model level stats, like 'model_r2'
        available_paras_this_model = [p for p in para_name_mapper if p in 
                                    df_this_model.columns[df_this_model.columns.get_level_values('stat_name') == stat
                                                        ].get_level_values('var_name').unique()]
        if available_paras_this_model:
            _, para = select_para(multi=False,
                                       suffix=suffix,
                                        available_paras=available_paras_this_model,
                                        default_paras=available_paras_this_model[0] if default_paras is None else default_paras,
                                        label='which variable',
                                        col=cols[1])
        else:
            para = ''
        
        column_to_map = (period, model, stat, para)
        column_to_map_name = f'{stat}_{para_name_mapper[para]}' if para != '' else stat
        if_map_pure = 'pure' in column_to_map[2]
        
    elif type_to_map == 'unit stats':
        column_to_map = st.selectbox("which stat", 
                                    ss.ccf_stat_names, 
                                    index=ss.ccf_stat_names.index('avg_firing_rate'),
                                    key=f'stat{suffix}')
        column_to_map_name = column_to_map
        if_map_pure = False
    
    column_selected = ss.df_unit_filtered_merged.loc[:, [column_to_map]]
    column_selected.dropna(inplace=True)
    
    if_take_abs = st.checkbox("Use abs()?", value=False, key='abs'+suffix) if np.any(column_selected < 0) else False

    if if_take_abs:
        column_selected = np.abs(column_selected)
        
    return dict(column_to_map=column_to_map, column_to_map_name=column_to_map_name, 
                if_map_pure=if_map_pure, if_take_abs=if_take_abs,
                column_selected=column_selected)
    

def add_unit_filter():
    with st.expander("Unit filter", expanded=True):   
        ss.df_unit_filtered = filter_dataframe(df=ss.df['df_ephys_units'])
        # Join with df_period_linear_fit_all here! (A huge dataframe with all things merged (flattened multi-level columns)
        ss.df_unit_filtered_merged = ss.df_unit_filtered.set_index(ss.unit_key_names + ['area_of_interest']
                                                                                        ).join(ss.df['df_period_linear_fit_all'], how='inner')
        
        n_units = len(ss.df_unit_filtered)
        n_animal = len(ss.df_unit_filtered['subject_id'].unique())
        n_insertion = len(ss.df_unit_filtered.groupby(['subject_id', 'session', 'insertion_number']))
        st.markdown(f'#### {n_units} units, {n_animal} mice, {n_insertion} insertions')
        

def add_unit_selector():
    with st.expander(f'Unit selector', expanded=True):
        
        n_units = len(ss.df_unit_filtered)
                        
        with st.expander(f"Filtered: {n_units} units", expanded=False):
            st.dataframe(ss.df_unit_filtered)
        
        for i, source in enumerate(ss.select_sources):
            df_selected_this = ss[f'df_selected_from_{source}']
            cols = st.columns([4, 1])
            with cols[0].expander(f"Selected: {len(df_selected_this)} units from {source}", expanded=False):
                st.dataframe(df_selected_this)
                
            if cols[1].button('❌' + ' '*i):  # Avoid duplicat key
                ss[f'df_selected_from_{source}'] = pd.DataFrame(columns=[ss.unit_key_names])
                st.experimental_rerun()
        
        # Sync selected units across sources
        cols = st.columns([4, 1])
        sync_using = cols[0].selectbox('Sync all using', ss.select_sources, index=0)
        cols[1].write('\n\n')
        if cols[1].button('🔄'):
            for source in ss.select_sources:
                if source != sync_using:
                    ss[f'df_selected_from_{source}'] = ss[f'df_selected_from_{sync_using}']
            st.experimental_rerun()
                