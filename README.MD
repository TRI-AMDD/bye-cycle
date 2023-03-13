# 🔋 Bye-Cycle 🔋

[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/TRI-AMDD/bye-cycle)
![tests](https://github.com/TRI-AMDD/bye-cycle/actions/workflows/test.yml/badge.svg)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


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

[See paper](https://chemrxiv.org/engage/chemrxiv/public-dashboard) and the citation:



License
--------

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)



Authors
--------

`bye-cycle` is developed by [Mehrad Ansari](https://mehradans92.github.io/).

