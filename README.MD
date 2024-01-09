# 🔋 Bye-Cycle 🔋

[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/TRI-AMDD/bye-cycle)
![tests](https://github.com/TRI-AMDD/bye-cycle/actions/workflows/test.yml/badge.svg)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![License: CC BY-NC-ND 4.0](http://mirrors.creativecommons.org/presskit/buttons/88x31/svg/by-nc-nd.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![DOI:10.26434/chemrxiv-2023-sm0lj](https://zenodo.org/badge/DOI/10.26434/chemrxiv-2023-sm0lj.svg)](https://doi.org/10.26434/chemrxiv-2023-sm0lj)

<p align="center">
    <img width="650" src="https://user-images.githubusercontent.com/51170839/222996858-332337d7-493c-4600-a777-335111d94729.gif" alt="Bye-Cycle">
</p>




<p align="center"> An automated deep learning plugin for early battery degradation inference. </p>


Overview
--------
<p align="center">
    <img width="650" src="https://user-images.githubusercontent.com/51170839/224599359-24fea431-7cc0-417a-afe0-bf268795baf2.png" alt="Overview">
</p>

<p>
    Our recurrent neural network model inputs time-series data for current and voltage coming from a single, movable window of recent cycles, predicting near-term ($m_1$) and long-term ($m_2$) as slopes of the capacity fade curve in the consecutive prediction window. Approximately equal values of slopes (with small magnitudes) accounts for a healthy battery at early cycles, whereas diverging slopes show a likely failing battery at the vicinity of the knee onset. The resource-expensive training is done on cloud computers, however, the cheap-to-evaluate inference model can effectively run on low commodity CPU/GPUs, allowing for an on-board vehicle inference opportunity for cells in operation. Additionally, this approach can be insightful for qualifying used lithium-ion batteries with <strong>unknown cycling histories</strong> for second life applications.
</p>


Features
--------

`bye-cycle` uses a rolling window of cycles containing cycle-independant features (current and voltage time-series data) to make inference on the magnitude of the drop in cell's capacity. User gets to choose variations of input and output window sizes as a hyperparameter in reconstructing the discharge capacity profile.

<p align="center">
    <em><strong>Input window size:</strong> 50 &nbsp;  <strong>Output window size:</strong> 50</em>
    <img width="850" src="https://user-images.githubusercontent.com/106785874/184695706-06aade46-a02e-4882-b1b7-dbf6223faccd.gif" alt="discharge_curve.gif">
</p>
<p align="center">
    <em><strong>Input window size:</strong> 10 &nbsp;  <strong>Output window size:</strong> 100</em>
    <img width="850" src="https://user-images.githubusercontent.com/106785874/184691653-c812bc67-be79-464e-a096-90c5f8cbd119.gif" alt="discharge_curve.gif">
</p>


Installation
--------

```python
pip install bye-cycle@git+https://github.com/TRI-AMDD/bye-cycle.git
```

or you can clone the source code and install in developer mode:

```python
git clone https://github.com/TRI-AMDD/bye-cycle.git && cd bye-cycle
pip install -e .
```


Citation
--------

See [paper](https://www.sciencedirect.com/science/article/pii/S2352152X23036782) and the citation:

```bibtex
@article{ANSARI2024110279,
title = {History-agnostic battery degradation inference},
journal = {Journal of Energy Storage},
volume = {81},
pages = {110279},
year = {2024},
issn = {2352-152X},
doi = {https://doi.org/10.1016/j.est.2023.110279},
url = {https://www.sciencedirect.com/science/article/pii/S2352152X23036782},
author = {Mehrad Ansari and Steven B. Torrisi and Amalie Trewartha and Shijing Sun},
keywords = {Battery degradation, Used battery, Unknown cycling history, Deep learning, Second life applications},
abstract = {Lithium-ion batteries (LIBs) have attracted widespread attention as an efficient energy storage device on electric vehicles (EV) to achieve emission-free mobility. However, the performance of LIBs deteriorates with time and usage, and the state of health of used batteries are difficult to quantify. Having accurate estimations of a battery’s remaining life across different life stages would benefit maintenance, safety, and serve as a means of qualifying used batteries for second-life applications. Since the full history of a battery may not always be available in downstream applications, in this study, we demonstrate a deep learning framework that enables dynamic degradation rate prediction, including both short-term and long-term forecasting, while requiring only the most recent battery usage information. Specifically, our model takes a rolling window of current and voltage time-series inputs, and predicts the near-term and long-term capacity fade via a recurrent neural network. We exhaustively benchmark our model against a naive extrapolating model by evaluating the error on reconstructing the discharge capacity profile under different settings. We show that our model’s performance in accurately inferring the battery’s degradation profile is agnostic with respect to cell cycling history and its current state of health. This approach can provide a promising path towards evaluating battery health in running vehicles, enhance edge-computing battery diagnostics, and determine the state of health for used batteries with unknown cycling histories.}
}
```


License
--------

[![License: CC BY-NC-ND 4.0](http://mirrors.creativecommons.org/presskit/buttons/88x31/svg/by-nc-nd.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)


Authors
--------

`bye-cycle` is developed by [Mehrad Ansari](https://mehradans92.github.io/).

