import plotly
import plotly.graph_objects as go

def add_plotly_errorbar(x, y, err, color, fig, alpha=0.2, name='', 
                        mode=None,
                        legend_group=None, subplot_specs=None, **kwargs):
    if legend_group is None:
        legend_group = f'group_{name}'
        
    valid_y = y.notna()
    y = y[valid_y]
    x = x[valid_y]
    err = err[valid_y]
    err[~err.notna()] = 0
    
    fig.add_trace(go.Scattergl(    
        x=x, 
        y=y, 
        # error_y=dict(type='data',
        #             symmetric=True,
        #             array=tuning_sem),
        name=name,
        legendgroup=legend_group,
        mode="markers+lines" if mode is None else mode,        
        marker_color=color,
        opacity=1,
        **kwargs,
        ),
        **subplot_specs)
    
    fig.add_trace(go.Scatter(
            # name='Upper Bound',
            x=x,
            y=y + err,
            mode='lines',
            marker=dict(color=color),
            line=dict(width=0),
            legendgroup=legend_group,
            showlegend=False,
            hoverinfo='skip',
            ),
            **subplot_specs)
    
    fig.add_trace(go.Scatter(
                    # name='Upper Bound',
                    x=x,
                    y=y - err,
                    mode='lines',
                    marker=dict(color=color),
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor=f'rgba({plotly.colors.convert_colors_to_same_type(color)[0][0].split("(")[-1][:-1]}, {alpha})',
                    legendgroup=legend_group,
                    showlegend=False,
                    hoverinfo='skip'
                ),
                  **subplot_specs)  
