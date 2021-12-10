Adding Neural Architecture Search to Cerebro with Autokeras
=======

In this project, we adapt NAS module provided by Autokeras to a scalable setting which employs Cerebro as the underlying training backend. The system generates and explores the NAS search space using Autokeras. The created training tasks are consumed by Cerebro to meet the resource demand. Extensive experiments on real-world datasets have been conducted to estimate the effectiveness of search algorithm and the improvement in efficiency.
 
``Cerebro`` is a data system for optimized deep learning model selection. It uses a novel parallel execution strategy
called **Model Hopper Parallelism (MOP)** to execute end-to-end deep learning model selection workloads in a more 
resource-efficient manner. Detailed technical information about ``Cerebro`` can be found in our 
[Technical Report](https://adalabucsd.github.io/papers/TR_2020_Cerebro.pdf).

Documentation
-------------
Detailed documentation about the system can be found [here](https://adalabucsd.github.io/cerebro-system/).

Acknowledgement
---------------
This work is a project of the course CSE 234 of the University of California, San Diego. We are grateful for the help and input from Prof. Arun Kumar and the teaching assistant Yuhao Zhang. 

In addition, all of the distributed experiments were conducted with the CloudLab platform, we appreciate the computational resources.
