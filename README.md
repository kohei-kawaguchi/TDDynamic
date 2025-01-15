# Football Markov

## Requirements

- Python 3.10.8
- Poetry 1.8.2

## Installation

```bash
poetry install
```

# Singe-agent model

## Model

The state variable is $z_{it}$ following an AR(1) process
$$
z_{it} = \mu (1 - \alpha) + \alpha z_{i,t-1} + \nu_{it}.
$$

The payoff of firm $i$ in time $t$
$$
\pi_{it} = \beta z_{it} a_{it} - \lambda (1 - a_{i, t-1}) a_{it} + \sum_{a = 0}^1 1\{a_{it} = a\}\epsilon_{a it} = \pi(a_{it}, a_{i, t-1}, z_{it}) + \sum_{a = 0}^1 1\{a_{it} = a\}\epsilon_{a it},
$$
and $\epsilon_{it}$ follows an i.i.d. type-I extreme value distribution.

The action $a_{it}$ takes either a value of 0 (inactive) or 1 (active).

We discretize $z_{it}$ into $G$ grid points according to Tauchen (1986).

The state variable of firm $i$ in time $t$ is
$$
s_{it} = (a_{i, t - 1}, z_{it})',
$$
and there are $L = G \times 2$ points:
$$
\begin{pmatrix}
0 & z_1 \\
1 & z_1 \\
\vdots & \vdots \\
0 & z_G \\
1 & z_G 
\end{pmatrix}.
$$

The payoff matrix is
$$
\Pi = 
\begin{pmatrix}
\pi(0, 0, z_1) \\
\pi(1, 0, z_1) \\
\pi(0, 1, z_1) \\
\pi(1, 1, z_1) \\
\vdots \\
\pi(0, 0, z_G) \\
\pi(1, 0, z_G) \\
\pi(0, 1, z_G) \\
\pi(1, 1, z_G) \\
\end{pmatrix},
$$
and the transition matrix is
$$
G = 
\begin{pmatrix}
g(0, 0, z_1, 0, z_1) & g(0, 0, z_1, 1, z_1) & \dots & (0, 0, z_1, 0, z_G) & g(0, 0, z_1, 1, z_G)\\
g(1, 0, z_1, 0, z_1) & g(1, 0, z_1, 1, z_1) & \dots & g(1, 0, z_1, 0, z_G) & g(1, 0, z_1, 1, z_G)\\
g(0, 1, z_1, 0, z_1) & g(0, 1, z_1, 1, z_1) & \dots & g(0, 1, z_1, 0, z_G) & g(0, 1, z_1, 1, z_G)\\
g(1, 1, z_1, 0, z_1) & g(1, 1, z_1, 1, z_1) & \dots & g(1, 1, z_1, 0, z_G) & g(1, 1, z_1, 1, z_G)\\
\vdots & & & & \\
g(0, 0, z_G, 0, z_1) & g(0, 0, z_G, 1, z_1) & \dots & g(0, 0, z_G, 0, z_G) & g(0, 0, z_G, 1, z_G)\\ 
g(1, 0, z_G, 0, z_1) & g(1, 0, z_G, 1, z_1) & \dots & g(1, 0, z_G, 0, z_G) & g(1, 0, z_G, 1, z_G)\\
g(0, 1, z_G, 0, z_1) & g(0, 1, z_G, 1, z_1) & \dots & g(0, 1, z_G, 0, z_G) & g(0, 1, z_G, 1, z_G)\\
g(1, 1, z_G, 0, z_1) & g(1, 1, z_G, 1, z_1) & \dots & g(1, 1, z_G, 0, z_G) & g(1, 1, z_G, 1, z_G)\\
\end{pmatrix}.
$$

## Main files

- `reference/karun/simulate_single.py`: solve the model and simulate the data from the solution
- `reference/karun/estimate_single.py`: estimate the parameter using the simulated data

## Function definitions

- `td_dynamic/karun/utility.py`: utility functions
- `td_dynamic/karun/simulate_single.py`: functions to simulate the data from the solution
- `td_dynamic/karun/estimate_single_semi_gradient.py`: functions to estimate h and g using semi-gradient method
- `td_dynamic/karun/estimate_single_avi.py`: functions to estimate h and g using approximate value iteration method
- `td_dynamic/karun/estimate_single_objective.py`: functions to evaluate the log-likelihood and estimate parameters by maximizing the log-likelihood

## Test files

- `reference/karun/test_simulate_single.py`: test functions in `simulate_single.py`
- `reference/karun/test_estimate_single_semi_gradient.py`: test functions in `estimate_single_semi_gradient.py`
- `reference/karun/test_estimate_single_avi.py`: test functions in `estimate_single_avi.py`
- `reference/karun/test_estimate_single_objective.py`: test functions in `estimate_single_objective.py`

# Multiple-agent model

## Model

There are $N$ potential firms in a market.

The state variable of the market is $z_{t}$ following an AR(1) process
$$
z_{t} = \mu (1 - \alpha) + \alpha z_{t-1} + \nu_{t}.
$$

The payoff of firm $i$ in time $t$
$$
\pi_{it} = \beta z_{t} a_{it} - \alpha \sum_{j = 1}^N a_{jt} - \lambda (1 - a_{i, t-1}) a_{it} + \sum_{a = 0}^1 1\{a_{it} = a\}\epsilon_{a it} = \pi_i(a_t, a_{t - 1}, z_t) + \sum_{a = 0}^1 1\{a_{it} = a\}\epsilon_{a it},
$$
and $\epsilon_{it}$ follows an i.i.d. type-I extreme value distribution.

The action $a_{it}$ takes either a value of 0 (inactive) or 1 (active).

Therefore, the number of action profiles is $m_a = 2^N$.

We discretize $z_{t}$ into $G$ grid points according to Tauchen (1986).

The state variable of firm $i$ in time $t$ is
$$
s_{it} = (a_{t - 1}, z_{t})',
$$
and there are $L = G \times m_a$ points:
$$
\begin{pmatrix}
0 & z_1 \\
\vdots & \vdots \\
m_a & z_1 \\
\vdots & \vdots \\
0 & z_G \\
\vdots & \vdots \\
m_a & z_G 
\end{pmatrix}.
$$

The payoff matrix is
$$
\begin{split}
\Pi_i &= 
\begin{pmatrix}
\pi_i(0, 0, z_1) \\
\vdots \\
\pi_i(m_a, 0, z_1) \\
\vdots \\
\pi_i(0, m_a, z_1) \\
\vdots \\
\pi_i(m_a, m_a, z_1) \\
\vdots \\
\pi_i(0, 0, z_G) \\
\vdots \\
\pi_i(m_a, 0, z_G) \\
\vdots \\
\pi_i(0, m_a, z_G) \\
\vdots \\
\pi_i(m_a, m_a, z_G) \\
\vdots \\
\vdots \\
\pi_i(m_a, m_a, z_G) \\
\end{pmatrix}\\
&=
\begin{pmatrix}
\Pi_i(0, z_1) \\
\vdots \\
\Pi_i(m_a, z_1) \\
\vdots \\
\Pi_i(0, z_G) \\
\vdots \\
\Pi_i(m_a, z_G) \\
\end{pmatrix}\\
&=
\begin{pmatrix}
\Pi_i(z_1) \\
\vdots \\
\Pi_i(z_G) 
\end{pmatrix}.
\end{split}
$$
and the transition matrix is
$$
\begin{split}
G &= 
\begin{pmatrix}
g(0, 0, z_1, 0, z_1) & \dots & g(0, 0, z_1, m_a, z_1) & \dots & g(0, 0, z_1, 0, z_G) & \dots & g(0, 0, z_1, m_a, z_G)\\
\vdots & \vdots & \cdots & \vdots & \vdots & \vdots & \vdots \\
g(m_a, 0, z_1, 0, z_1) & \dots & g(m_a, 0, z_1, m_a, z_1) & \dots & g(m_a, 0, z_1, 0, z_G) & \dots & g(m_a, 0, z_1, m_a, z_G)\\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
g(0, m_a, z_1, 0, z_1) & \dots & g(0, m_a, z_1, m_a, z_1) & \dots & g(0, m_a, z_1, 0, z_G) & \dots & g(0, m_a, z_1, m_a, z_G)\\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
g(m_a, m_a, z_1, 0, z_1) & \dots & g(m_a, m_a, z_1, m_a, z_1) & \dots & g(m_a, m_a, z_1, 0, z_G) & \dots & g(m_a, m_a, z_1, m_a, z_G)\\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
g(0, 0, z_G, 0, z_1) & \dots & g(0, 0, z_G, m_a, z_1) & \dots & g(0, 0, z_G, 0, z_G) & \dots & g(0, 0, z_G, m_a, z_G)\\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
g(m_a, 0, z_G, 0, z_1) & \dots & g(m_a, 0, z_G, m_a, z_1) & \dots & g(m_a, 0, z_G, 0, z_G) & \dots & g(m_a, 0, z_G, m_a, z_G)\\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
g(0, m_a, z_G, 0, z_1) & \dots & g(0, m_a, z_G, m_a, z_1) & \dots & g(0, m_a, z_G, 0, z_G) & \dots & g(0, m_a, z_G, m_a, z_G)\\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
g(m_a, m_a, z_G, 0, z_1) & \dots & g(m_a, m_a, z_G, m_a, z_1) & \dots & g(m_a, m_a, z_G, 0, z_G) & \dots & g(m_a, m_a, z_G, m_a, z_G)\\
\end{pmatrix}\\
&=
\begin{pmatrix}
G(0, z_1, 0, z_1) & \dots & G(0, z_1, m_a, z_1) & \dots & G(0, z_1, 0, z_G) & \dots & G(0, z_1, m_a, z_G)\\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
G(m_a, z_1, 0, z_1) & \dots & G(m_a, z_1, m_a, z_1) & \dots & G(m_a, z_1, 0, z_G) & \dots & G(m_a, z_1, m_a, z_G)\\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
G(0, z_G, 0, z_1) & \dots & G(0, z_G, m_a, z_1) & \dots & G(0, z_G, 0, z_G) & \dots & G(0, z_G, m_a, z_G)\\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
G(m_a, z_G, 0, z_1) & \dots & G(m_a, z_G, m_a, z_1) & \dots & G(m_a, z_G, 0, z_G) & \dots & G(m_a, z_G, m_a, z_G)\\
\end{pmatrix}\\
&=
\begin{pmatrix}
G(z_1, 0, z_1) & \dots & G(z_1, m_a, z_1) & \dots & G(z_1, 0, z_G) & \dots & G(z_1, m_a, z_G)\\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
G(z_G, 0, z_1) & \dots & G(z_G, m_a, z_1) & \dots & G(z_G, 0, z_G) & \dots & G(z_G, m_a, z_G)\\
\end{pmatrix}\\
&=
\begin{pmatrix}
G(z_1) \\
\vdots \\
G(z_G) 
\end{pmatrix}.
\end{split}
$$

## Main files

- `reference/karun/simulate_multiple.py`: solve the model and simulate the data from the solution
- `reference/karun/estimate_multiple.py`: estimate the parameter using the simulated data

## Function definitions

- `td_dynamic/karun/utility.py`: utility functions
- `td_dynamic/karun/simulate_multiple.py`: functions to simulate the data from the solution
- `td_dynamic/karun/estimate_multiple_semi_gradient.py`: functions to estimate h and g using semi-gradient method
- `td_dynamic/karun/estimate_multiple_avi.py`: functions to estimate h and g using approximate value iteration method
- `td_dynamic/karun/estimate_multiple_objective.py`: functions to evaluate the log-likelihood and estimate parameters by maximizing the log-likelihood

## Manualy test files

- `reference/karun/test_simulate_multiple.py`: test functions in `simulate_multiple.py`
- `reference/karun/test_estimate_multiple_semi_gradient.py`: test functions in `estimate_multiple_semi_gradient.py`
- `reference/karun/test_estimate_multiple_avi.py`: test functions in `estimate_multiple_avi.py`
- `reference/karun/test_estimate_multiple_objective.py`: test functions in `estimate_multiple_objective.py`


## Pytest files

- `reference/karun/pytest_simulate_multiple.py`: Pytest functions in `simulate_multiple.py`
- `reference/karun/pytest_estimate_multiple_semi_gradient.py`: Pytest functions in `estimate_multiple_semi_gradient.py`
- `reference/karun/pytest_estimate_multiple_avi.py`: Pytest functions in `estimate_multiple_avi.py`
- `reference/karun/pytest_estimate_multiple_objective.py`: Pytest functions in `estimate_multiple_objective.py`