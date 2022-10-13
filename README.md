README author: Bijaya Adhikari.
Dated: Januray 17, 2017. 


This is a reference implementation of the temporaledgecover/temporalnodecover algorithm described in the following paper.
Near-Optimal Spectral Disease Mitigation in Healthcare Facilities
Masahiro Kiji, D. M. Hasibul Hasan, Alberto M. Segre, Sriram V. Pemmaraju, and Bijaya Adhikari
In 2022 IEEE International Conference on Data Mining (ICDM 2022), Orlando, Florida.


Given a temporal biparitie network, temporalEdgeCOVER/temporalNodeCover returns a set of edges/nodes to remove from the input network which minimizes the spectral radius of the resulting graph.
Please check the paper for more detail.

==========================================================================================================================================================================================


### Temporal Edge Cover Demo Codes

This folder contains demo codes of Temporal Edge Cover.

##### language
* Python 3.8.3

##### libraries
* NumPy 1.19.
* SciPy 1.5.2
* NetworkX 2.5

##### Demos

###### generate a node/edge set to delete
* run demo_temp_edge_cover.py

###### compute lambda of the system matrix w/ or w/o deletion
* run demo_lambda.py
