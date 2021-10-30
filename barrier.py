import os
import functools
import numpy as np
import scipy.constants as const
import plotly.graph_objects as go

def classic_barrier(alpha, beta):
    a = 1 / (4 * alpha * np.abs(alpha - 1))
    b = beta * np.sqrt(np.abs(alpha - 1))
    if (alpha >= 0) and (alpha < 1):
        return 1 / (1 + a * np.sinh(b)**2)
    elif (alpha >= 1):
        return 1 / (1 + a * np.sin(b)**2)
    else:
        return np.nan

def quasi_classic_barrier(alpha, beta):
    return np.exp(-2 * beta * np.sqrt(np.abs(alpha - 1)))

if __name__ == '__main__':
    alphas = np.arange(0, 10, 0.001)
    betas = np.arange(1, 11, 1)
    start_beta = 1

    fig = go.Figure()
    for beta in betas:
        bquasi = np.array(list(map(functools.partial(quasi_classic_barrier, beta=beta), alphas)))
        fig.add_trace(go.Scatter(
            visible=False,
            x=alphas,
            y=bquasi,
            name='quasi-classic',
            hovertext=r'$\beta$='+str(beta),
            mode='lines',
            line=dict(color='red', width=2, dash='solid')
        ))

        bclassic = np.array(list(map(functools.partial(classic_barrier, beta=beta), alphas)))
        fig.add_trace(go.Scatter(
            visible=False,
            x=alphas,
            y=bclassic,
            name='classic',
            hovertext=r'$\beta$=' + str(beta),
            mode='lines',
            line=dict(color='blue', width=2, dash='solid')
        ))

    fig.data[start_beta * 2].visible = True
    fig.data[start_beta * 2 + 1].visible = True

    fig.update_traces(showlegend=True)
    fig.update_xaxes(range=[0, 5])
    fig.update_yaxes(range=[0, 1.5])
    fig.update_layout(
        title='Transmission coefficient',
        xaxis_title=r'$\alpha$',
        yaxis_title='D',
    )

    steps = []
    for i in range(len(fig.data) // 2):
        step = dict(
            method='update',
            args=[{'visible': [False] * len(fig.data)}],
            label=f'{i+1}'
        )
        step["args"][0]["visible"][i * 2] = True
        step["args"][0]["visible"][i * 2 + 1] = True
        steps.append(step)

    sliders = [dict(
        active=start_beta,
        currentvalue={"prefix": "beta = "},
        pad={"t": 30},
        steps=steps
    )]

    fig.update_layout(sliders=sliders)
    fig.show()
