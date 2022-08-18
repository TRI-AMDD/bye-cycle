![tests](https://github.com/TRI-AMDD/bye-cycle/actions/workflows/test.yml/badge.svg)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

<!---[![GitHub Repo Size](https://img.shields.io/github/repo-size/TRI-AMDD/bye-cycle?label=Repo+Size)](https://github.com/TRI-AMDD/bye-cycle/graphs/contributors)-->

<p align="center">
    <img width="300" src="https://user-images.githubusercontent.com/106785874/184673860-5d978d8d-4cb8-4aaf-bacb-79ffd15143a9.png" alt="Bye-Cycle">
</p>
<p align="center">
    <img width="460" src="https://user-images.githubusercontent.com/106785874/184672636-619480c6-3da1-464a-9617-0ac4dd29d22e.gif" alt="Logo">
</p>

<p align="center"> An automated deep learning plugin for early battery life-cycle and degradation inference. </p>


Features
--------

`bye-cycle` uses a rolling window of cycles containing cycle-independant features (current and voltage time-series data) to make inference on the magnitude of the drop in cell's capacity. User gets to choose variations of input and output window sizes as a hyperparameter in reconstructing the discharge capacity trajectory.

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

```
pip install bye-cycle@git+https://github.com/TRI-AMDD/bye-cycle.git
```
or you can clone the source code and install in developer mode:
```
git clone https://github.com/TRI-AMDD/bye-cycle.git && cd bye-cycle
pip install -e .
```

License
--------
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Authors
--------

`bye-cycle` is developed by [Mehrad Ansari](https://mehradans92.github.io/).
