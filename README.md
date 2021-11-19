Cerebro
=======
 
``Cerebro`` is a data system for optimized deep learning model selection. It uses a novel parallel execution strategy
called **Model Hopper Parallelism (MOP)** to execute end-to-end deep learning model selection workloads in a more 
resource-efficient manner. Detailed technical information about ``Cerebro`` can be found in our 
[Technical Report](https://adalabucsd.github.io/papers/TR_2020_Cerebro.pdf).


Run on CloudLab
-------
#### 1. Set up the environment

You need to manually upload the profile.py to create a Profile on CloudLab, since the file profile.py will cause
contradiction with PySpark and cause ImportError, I did not put it in this repository.

When creating an experiment on CloudLab, a node with hardware type that has higher disk size and memory size is more ideal. 
But it is not always possible to get an available one. Check [cluster status](https://www.cloudlab.us/resinfo.php) here

After entering the shell of a node:
 
```bash
git clone https://github.com/jiange91/cerebro-system.git
cd cerebro-system
```
Since the `bootstrap.sh` script is completed under Windows OS, need to manually change the file format.

```bash
sed -i "s/\r//" bootstrap.sh
```

Then run the script to set up the environment
```bash
sudo chmod 777 bootstrap.sh
sudo ./bootstrap.sh
```

Note that please do not leave the `sudo` when execute the `bootstrap.sh`.

Install
-------

The best way to install the ``Cerebro`` is via pip.

    pip install -U cerebro-dl

Alternatively, you can git clone and run the provided Makefile script

    git clone https://github.com/ADALabUCSD/cerebro-system.git && cd cerebro-system && make

You MUST be running on **Python >= 3.6** with **Tensorflow >= 2.2** and **Apache Spark >= 2.4**


Documentation
-------------

Detailed documentation about the system can be found [here](https://adalabucsd.github.io/cerebro-system/).


Acknowledgement
---------------
This project was/is supported in part by a Hellman Fellowship, the NIDDK of the NIH under award number R01DK114945, and an NSF CAREER Award.

We used the following projects when building Cerebro.
- [Horovod](https://github.com/horovod/horovod): Cerebro's Apache Spark implementation uses code from the Horovod's
 implementation for Apache Spark.
- [Petastorm](https://github.com/uber/petastorm): We use Petastorm to read Apache Parquet data from remote storage
 (e.g., HDFS)  
 
Publications
------------
If you use this software for research, plase cite the following papers:

```latex
@inproceedings{nakandala2019cerebro,
  title={Cerebro: Efficient and Reproducible Model Selection on Deep Learning Systems},
  author={Nakandala, Supun and Zhang, Yuhao and Kumar, Arun},
  booktitle={Proceedings of the 3rd International Workshop on Data Management for End-to-End Machine Learning},
  pages={1--4},
  year={2019}
}

```
