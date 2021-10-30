import os
import functools
import numpy as np
import scipy.constants as const
from scipy.special import hermite
from scipy.integrate import quad
import plotly.graph_objects as go

def c_div_momentum(n, x):
    if (np.abs(x) == 1):
        return 1
    return 1/np.sqrt(np.abs(np.pi * const.hbar * (n + 1/2) * (1 - x**2))) / 1

def wave_quasi_classical(eps, a, b, n, norm, skip_nan, x):
    x /= np.sqrt(2 * n + 1)
    if (x <= a-eps):
        return (-1)**n / 2 * c_div_momentum(n, x) * np.exp(-(n + 1/2) * (- x * np.sqrt(x**2 - 1) - np.arccosh(-x))) / norm
    elif (x > a+eps) and (x < b-eps):
        return c_div_momentum(n, x) * np.cos((n + 1/2) * (np.pi/2 - np.arcsin(x) - x*np.sqrt(1 - x**2)) - np.pi/4) / norm
    elif (x >= b+eps):
        return 1/2 * c_div_momentum(n, x) * np.exp(-(n + 1/2) * (x * np.sqrt(x**2 - 1) - np.arccosh(x))) / norm
    else:
        return 0 if skip_nan else np.nan

def prob_wave_quasi_classical(eps, a, b, n, x, norm, skip_nan):
    return wave_quasi_classical(eps, a, b, n, x, norm, skip_nan) ** 2

def N(n, x):
    return 1/np.sqrt(np.sqrt(np.pi) * 2**n * np.math.factorial(n)) * np.exp(- x**2 / 2)

def wave_precise(n, x):
    Hn = hermite(n)
    return Hn(x) * N(n, x)

def prob_wave_precise(n, x):
    return wave_precise(n, x) ** 2

if __name__ == '__main__':
    eps = 0.001
    a, b = -1, 1
    start_n, max_n = 5, 10
    xstart, xstop, xstep = -10, 10, 0.001
    x_normed = np.arange(xstart, xstop+xstep, xstep)

    fig = go.Figure()
    for n in np.arange(0, max_n+1, 1):
        wf_precise = wave_precise(n, x_normed)
        wf_int, err = quad(functools.partial(prob_wave_precise, n), -100, 100)
        # print(wf_int, err)
        wf_precise /= np.sqrt(wf_int)
        fig.add_trace(go.Scatter(
            visible=False,
            x=x_normed,
            y=wf_precise,
            name='precise',
            hovertext=f'n={n}',
            mode='lines',
            line=dict(color='blue', width=2, dash='solid')
        ))

        wf_quasi = np.array(list(map(functools.partial(wave_quasi_classical, eps, a, b, n, 1, False), x_normed)))
        wf_int, err = quad(functools.partial(prob_wave_quasi_classical, eps*100, a, b, n, 1, True), -100, 100)
        # print(wf_int, err)
        wf_quasi /= np.sqrt(wf_int)

        # Check norm
        # wf_int, err = quad(functools.partial(prob_wave_quasi_classical, 0, a, b, n, np.sqrt(wf_int)), -100, 100)
        # print(wf_int, err)
        # wf_quasi /= np.sqrt(wf_int)

        fig.add_trace(go.Scatter(
            visible=False,
            x=x_normed,
            y=wf_quasi,
            name='quasi-classical approx',
            hovertext=f'n={n}',
            mode='lines',
            line=dict(color='red', width=2, dash='solid')
        ))

    fig.data[start_n * 2].visible = True
    fig.data[start_n * 2 + 1].visible = True

    fig.update_traces(showlegend=True)
    fig.update_yaxes(range=[-6, 6])
    fig.update_layout(
        title='Wave functions of harmonic oscillator',
        xaxis_title='x',
        yaxis_title='\u03A8(x)',
    )

    steps = []
    for i in range(len(fig.data) // 2):
        step = dict(
            method='update',
            args=[{'visible': [False] * len(fig.data)}],
            label=f'{i}'
        )
        step["args"][0]["visible"][i * 2] = True
        step["args"][0]["visible"][i * 2 + 1] = True
        steps.append(step)

    sliders = [dict(
        active=start_n,
        currentvalue={"prefix": "n = "},
        pad={"t": 30},
        steps=steps
    )]

    fig.update_layout(sliders=sliders)
    fig.show()
