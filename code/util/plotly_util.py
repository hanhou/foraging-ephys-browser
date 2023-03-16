import plotly
import plotly.graph_objects as go

def add_plotly_errorbar(x, y, err, col, fig, alpha=0.2, name='', legend_group=None, **kwargs):
    if legend_group is None:
        legend_group = f'group_{name}'
    
    fig.add_trace(go.Scattergl(    
        x=x, 
        y=y, 
        # error_y=dict(type='data',
        #             symmetric=True,
        #             array=tuning_sem),
        name=name,
        legendgroup=legend_group,
        mode="markers+lines",        
        marker_color=col,
        opacity=1,
        **kwargs,
        ))
    
    fig.add_trace(go.Scatter(
            # name='Upper Bound',
            x=x,
            y=y + err,
            mode='lines',
            marker=dict(color=col),
            line=dict(width=0),
            legendgroup=legend_group,
            showlegend=False,
            hoverinfo='skip',
        ))
    fig.add_trace(go.Scatter(
                    # name='Upper Bound',
                    x=x,
                    y=y - err,
                    mode='lines',
                    marker=dict(color=col),
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor=f'rgba({plotly.colors.convert_colors_to_same_type(col)[0][0].split("(")[-1][:-1]}, {alpha})',
                    legendgroup=legend_group,
                    showlegend=False,
                    hoverinfo='skip'
                ))  
