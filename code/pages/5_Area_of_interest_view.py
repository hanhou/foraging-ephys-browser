import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from Home import add_unit_filter, get_fig_unit_all_in_one, init, select_t_sign_level

import importlib
xy_view = importlib.import_module('.3_X-Y_view', package='pages')


pure_unit_color_mapping =  {'pure_dQ': 'darkviolet',
                            'pure_sumQ': 'deepskyblue',
                            'pure_contraQ': 'darkblue',
                            'pure_ipsiQ': 'darkorange'}

sig_prop_color_mapping =  {'dQ': 'darkviolet',
                            'sumQ': 'deepskyblue',
                            'contraQ': 'darkblue',
                            'ipsiQ': 'darkorange',
                            'rpe': 'gray'}




@st.cache_data(ttl=24*3600)
def _polar_histogram(df_this_aoi, x_name, y_name, polar_method, bins, sign_level):

    df_sig = df_this_aoi.query(f'abs({x_name}) >= {sign_level} or abs({y_name}) >= {sign_level}')
    theta, r = _to_theta_r(df_sig[x_name], df_sig[y_name])
    weight = r if 'weighted' in polar_method else np.ones_like(theta)

    counts, _ = np.histogram(a=theta, bins=bins, weights=weight)
    
    if 'in all neurons' in polar_method:
        return counts / len(df_this_aoi) * 100 # Not sum to 1
    elif 'in significant neurons' in polar_method:
        return counts / len(df_sig) * 100 # Sum to 1
    else:
        return counts / np.sum(counts)  # weighted r


@st.cache_data(ttl=24*3600)        
def plot_polar(df_unit_filtered, x_name, y_name, polar_method, n_bins, if_errorbar, sign_level):
        
    bins = np.linspace(-np.pi, np.pi, num=n_bins + 1)
    bin_center = np.rad2deg(np.mean([bins[:-1], bins[1:]], axis=0)) 

    polar_hist = df_unit_filtered.groupby('area_of_interest').apply(lambda df_this_aoi: _polar_histogram(df_this_aoi, x_name, y_name, polar_method, bins, sign_level))

    fig = go.Figure() 
    
    for aoi in st.session_state.df['aoi'].index:
        if aoi not in df_unit_filtered.area_of_interest.values:
            continue
        
        hist = polar_hist[aoi]
        fig.add_trace(go.Scatterpolar(r=np.hstack([hist, hist[0]]),
                                        theta=np.hstack([bin_center, bin_center[0]]),
                                        mode='lines + markers',
                                        marker_color=st.session_state.aoi_color_mapping[aoi], 
                                        legendgroup=aoi,
                                        name=aoi,
                    )
                    #   color_discrete_sequence=px.colors.sequential.Plasma_r,
                    #   template="plotly_dark",
                    )
        
        # add binomial errorbar
        if 'in all neurons' in polar_method and if_errorbar:
            n = len(df_unit_filtered.query(f'area_of_interest == "{aoi}"')) 
            for p, theta in zip(hist, bin_center):
                ci_95 = 1.96 * np.sqrt(p * (100 - p) / n)
                fig.add_trace(go.Scatterpolar(r=[p - ci_95, p + ci_95],
                                              theta=[theta, theta],
                                              mode='lines',
                                              marker_color=st.session_state.aoi_color_mapping[aoi],
                                              legendgroup=aoi,
                                              name=aoi,
                                              showlegend=False,
                                              ))
    
    # add type I error
    fig.add_trace(go.Scatterpolar(r=np.full(len(bin_center) + 1, 0.01 * 100 / n_bins),
                                  theta=np.hstack([bin_center, bin_center[0]]),
                                  mode='lines',
                                  line=dict(color='black', dash='dot'),
                                  name='Type I error'
                                  )) # Type I error divided by 4
    
    fig.update_layout(height=800, width=800, font=dict(size=20))
    return fig



def add_xy_selector():
    with st.expander("Select axes", expanded=True):
        with st.form("axis_selection"):
            col3, col4 = st.columns([1, 1])
            with col3:
                x_name = st.selectbox("x axis", st.session_state.scatter_stats_names, index=st.session_state.scatter_stats_names.index('t_dQ_iti'))
            with col4:
                y_name = st.selectbox("y axis", st.session_state.scatter_stats_names, index=st.session_state.scatter_stats_names.index('t_sumQ_iti'))
            st.form_submit_button("update axes")
    return x_name, y_name


def _to_theta_r(x, y):
    return np.arctan2(y, x), np.sqrt(x**2 + y**2)
    

def app():
    with st.sidebar:
        add_unit_filter()
        select_t_sign_level()
   
    # -- axes selector --
    x_name, y_name = add_xy_selector()

    # -- scatter --
    col1, col2 = st.columns((1, 1))
    with col1:  # Raw scatter
        # selected_points_scatter = xy_view.add_scatter_return_selected(st.session_state.df_unit_filtered, x_name, y_name)
        pass
        
    with col2: # Polar distribution
        if len(st.session_state.df_unit_filtered):   
            col21, col22, col23 = st.columns((1,1,1))
            with col21:
                n_bins = st.slider("number of polar bins", 4, 32, 16, 4)
            with col22:
                polar_method = st.selectbox(label="polar method", options=['proportion in all neurons', 
                                                                        'proportion in significant neurons',
                                                                        'significant units weighted by r'],
                                            index=0)
            with col23:
                if_errorbar = st.checkbox("binomial 95% CI", value=True, disabled='in all neurons' not in polar_method)
                        
            fig = plot_polar(st.session_state.df_unit_filtered, x_name, y_name, polar_method, n_bins, if_errorbar, 
                             st.session_state['t_sign_level'])

            # # Select other Plotly events by specifying kwargs
            # selected_points_scatter = plotly_events(fig, click_event=True, hover_event=False, select_event=False,
            #                                         override_height=800, override_width=800)
            st.plotly_chart(fig, use_container_width=True)
        pass
    
                

    container_unit_all_in_one = st.container()
    
    with container_unit_all_in_one:
        # with st.expander("Expand to see all-in-one plot for selected unit", expanded=True):
        if len(st.session_state.selected_points) == 1:  # Priority to select on scatter plot
            key = st.session_state.aggrid_outputs['data'].query(f'{x_name} == {st.session_state.selected_points[0]["x"]} and {y_name} == {st.session_state.selected_points[0]["y"]}')
            if len(key):
                unit_fig = get_fig_unit_all_in_one(dict(key.iloc[0]))
                st.image(unit_fig, output_format='PNG', width=3000)

        # elif len(st.session_state.aggrid_outputs['selected_rows']) == 1:
        #     unit_fig = get_fig_unit_all_in_one(st.session_state.aggrid_outputs['selected_rows'][0])
        #     st.image(unit_fig, output_format='PNG', width=3000)

     
    # if selected_points_scatter and selected_points_scatter != st.session_state.selected_points:
    #     st.session_state.selected_points = selected_points_scatter
    #     st.experimental_rerun()
        
if __name__ == '__main__':
    if 'df' not in st.session_state: 
        init()

    app()