<h1 align="center">Federated Learning with Differential Privacy RP2 </h1>
<div align="center"> 

</div>


[//]: # (This repository collects related papers and corresponding codes on DP-based FL.)

[//]: # (## Code)

## TODO 


[//]: # (Tip: the code of this repository is my personal implementation, if there is an inaccurate place please contact me, welcome to discuss with each other. The FL code of this repository is based on this [repository]&#40;https://github.com/wenzhu23333/Federated-Learning&#41; .I hope you like it and support it. Welcome to submit PR to improve the  repository.)

[//]: # ()
[//]: # (Note that in order to ensure that each client is selected a fixed number of times &#40;to compute privacy budget each time the client is selected&#41;, this code uses round-robin client selection, which means that each client is selected sequentially.)

[//]: # ()
[//]: # (Important note: The number of FL local update rounds used in this code is all 1, please do not change, once the number of local iteration rounds is changed, the sensitivity in DP needs to be recalculated, the upper bound of sensitivity will be a large value, and the privacy budget consumed in each round will become a lot, so please use the parameter setting of Local epoch = 1.)

[//]: # ()
[//]: # (### Parameter List)

[//]: # ()
[//]: # (**Datasets**: MNIST, Cifar-10, FEMNIST, Fashion-MNIST, Shakespeare.)

[//]: # ()
[//]: # (**Model**: CNN, MLP, LSTM for Shakespeare)

[//]: # ()
[//]: # (**DP Mechanism**: Laplace, Gaussian&#40;Simple Composition&#41;, Gaussian&#40;*moments* accountant&#41;)

[//]: # ()
[//]: # (**DP Parameter**: $\epsilon$ and $\delta$)

[//]: # ()
[//]: # (**DP Clip**: In DP-based FL, we usually clip the gradients in training and the clip is an important parameter to calculate the sensitivity.)

[//]: # ()
[//]: # (### Example Results)

[//]: # ()
[//]: # (Experiments code:)

[//]: # ()
[//]: # (```shell)

[//]: # (pip3 install -r requirements.txt)

[//]: # (bash run.sh)

[//]: # (```)

[//]: # ()
[//]: # (Drawing code: )

[//]: # (```shell)

[//]: # (python3 draw.py)

[//]: # (```)

[//]: # ()
[//]: # (#### Gaussian &#40;Simple Composition&#41;)

[//]: # ()
[//]: # (![Mnist]&#40;mnist_gaussian.png&#41;)

[//]: # ()
[//]: # (#### Gaussian &#40;Moment Account&#41;)

[//]: # ()
[//]: # (![Mnist]&#40;mnist_gaussian_MA.png&#41;)

[//]: # ()
[//]: # (#### Laplace)

[//]: # ()
[//]: # (![Mnist]&#40;mnist_gaussian_laplace.png&#41;)

[//]: # ()
[//]: # ()
[//]: # (### No DP)

[//]: # ()
[//]: # (```shell)

[//]: # (python main.py --dataset mnist --model cnn --dp_mechanism no_dp)

[//]: # (```)

[//]: # (### Gaussian Mechanism)

[//]: # ()
[//]: # (#### Simple Composition)

[//]: # ()
[//]: # (Based on Simple Composition in DP. )

[//]: # ()
[//]: # (In other words, if a client's privacy budget is $\epsilon$ and the client is selected $T$ times, the client's budget for each noising is $\epsilon / T$.)

[//]: # ()
[//]: # (```shell)

[//]: # (python main.py --dataset mnist --model cnn --dp_mechanism Gaussian --dp_epsilon 10 --dp_delta 1e-5 --dp_clip 10)

[//]: # (```)

[//]: # ()
[//]: # (#### Moments Accountant)

[//]: # ()
[//]: # (We use [Tensorflow Privacy]&#40;https://github.com/tensorflow/privacy&#41; to calculate noise scale of the Moment Account&#40;MA&#41; for Gaussian Mechanism.)

[//]: # ()
[//]: # (```shell)

[//]: # (python main.py --dataset mnist --model cnn --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --dp_clip 10 --dp_sample 0.01)

[//]: # (```)

[//]: # (See the paper for detailed mechanism. )

[//]: # ()
[//]: # (Abadi, Martin, et al. "Deep learning with differential privacy." *Proceedings of the 2016 ACM SIGSAC conference on computer and communications security*. 2016.)

[//]: # ()
[//]: # (### Laplace Mechanism)

[//]: # ()
[//]: # (Based on Simple Composition in DP. )

[//]: # ()
[//]: # (```shell)

[//]: # (python main.py --dataset mnist --model cnn --dp_mechanism Laplace --dp_epsilon 30 --dp_clip 50)

[//]: # (```)

[//]: # ()
[//]: # ()
[//]: # (## Papers)

[//]: # ()
[//]: # (- Reviews)

[//]: # (  - Rodríguez-Barroso, Nuria, et al. "[Federated Learning and Differential Privacy: Software tools analysis, the Sherpa. ai FL framework and methodological guidelines for preserving data privacy.]&#40;https://www.sciencedirect.com/science/article/pii/S1566253520303213&#41;" *Information Fusion* 64 &#40;2020&#41;: 270-292.)

[//]: # (- Gaussian Mechanism)

[//]: # (  - Wei, Kang, et al. "[Federated learning with differential privacy: Algorithms and performance analysis.]&#40;https://ieeexplore.ieee.org/document/9069945&#41;" *IEEE Transactions on Information Forensics and Security* 15 &#40;2020&#41;: 3454-3469.)

[//]: # (  - Y. Zhou, et al.,"[Optimizing the Numbers of Queries and Replies in Convex Federated Learning with Differential Privacy]&#40;https://ieeexplore.ieee.org/document/10008087/&#41;" in IEEE Transactions on Dependable and Secure Computing, 2023.)

[//]: # (  - K. Wei, et al.,"[User-Level Privacy-Preserving Federated Learning: Analysis and Performance Optimization]&#40;https://ieeexplore.ieee.org/document/9347706&#41;" in IEEE Transactions on Mobile Computing, vol. 21, no. 09, pp. 3388-3401, 2022.)

[//]: # (  - Geyer, Robin C., Tassilo Klein, and Moin Nabi. "[Differentially private federated learning: A client level perspective.]&#40;https://arxiv.org/abs/1712.07557&#41;" *arXiv preprint arXiv:1712.07557* &#40;2017&#41;.)

[//]: # (  - Seif, Mohamed, Ravi Tandon, and Ming Li. "[Wireless federated learning with local differential privacy.]&#40;https://arxiv.org/abs/2002.05151&#41;" *2020 IEEE International Symposium on Information Theory &#40;ISIT&#41;*. IEEE, 2020.)

[//]: # (  - Mohammadi, Nima, et al. "[Differential privacy meets federated learning under communication constraints.]&#40;https://ieeexplore.ieee.org/document/9511628&#41;" *IEEE Internet of Things Journal* &#40;2021&#41;.)

[//]: # (  - Truex, Stacey, et al. "[A hybrid approach to privacy-preserving federated learning.]&#40;https://dl.acm.org/doi/10.1145/3338501.3357370&#41;" *Proceedings of the 12th ACM workshop on artificial intelligence and security*. 2019.)

[//]: # (  - Naseri, Mohammad, Jamie Hayes, and Emiliano De Cristofaro. "[Toward robustness and privacy in federated learning: Experimenting with local and central differential privacy.]&#40;https://arxiv.org/abs/2009.03561&#41;" *arXiv e-prints* &#40;2020&#41;: arXiv-2009.)

[//]: # (  - Malekzadeh, Mohammad, et al. "[Dopamine: Differentially private federated learning on medical data.]&#40;https://arxiv.org/abs/2101.11693&#41;" *arXiv preprint arXiv:2101.11693* &#40;2021&#41;.)

[//]: # (- Laplace Mechanism)

[//]: # (  - Wu, Nan, et al. "[The value of collaboration in convex machine learning with differential privacy.]&#40;https://www.computer.org/csdl/proceedings-article/sp/2020/349700a485/1j2LfLp7Sik&#41;" *2020 IEEE Symposium on Security and Privacy &#40;SP&#41;*. IEEE, 2020.)

[//]: # (  - Y. Zhou, et al.,"[Optimizing the Numbers of Queries and Replies in Convex Federated Learning with Differential Privacy]&#40;https://ieeexplore.ieee.org/document/10008087/&#41;" in IEEE Transactions on Dependable and Secure Computing, 2023.)

[//]: # (  - L. Cui, J. Ma, Y. Zhou and S. Yu, "[Boosting Accuracy of Differentially Private Federated Learning in Industrial IoT With Sparse Responses,]&#40;https://ieeexplore.ieee.org/document/9743613/&#41;" in IEEE Transactions on Industrial Informatics, 2023. )

[//]: # (  - Liu, Xiaoyuan, et al. "[Adaptive privacy-preserving federated learning.]&#40;https://link.springer.com/article/10.1007/s12083-019-00869-2&#41;" *Peer-to-Peer Networking and Applications* 13.6 &#40;2020&#41;: 2356-2366.)

[//]: # (  - Zhao, Yang, et al. "[Local differential privacy-based federated learning for internet of things.]&#40;https://ieeexplore.ieee.org/document/9253545/&#41;" *IEEE Internet of Things Journal* 8.11 &#40;2020&#41;: 8836-8853.)

[//]: # (  - Fu, Yao, et al. "[On the practicality of differential privacy in federated learning by tuning iteration times.]&#40;https://arxiv.org/abs/2101.04163&#41;" *arXiv preprint arXiv:2101.04163* &#40;2021&#41;.)

[//]: # (- Other Mechanism)

[//]: # (  - Zhao, Yang, et al. "[Local differential privacy-based federated learning for internet of things.]&#40;https://ieeexplore.ieee.org/document/9253545/&#41;" *IEEE Internet of Things Journal* 8.11 &#40;2020&#41;: 8836-8853.)

[//]: # (  - Truex, Stacey, et al. "[LDP-Fed: Federated learning with local differential privacy.]&#40;https://dl.acm.org/doi/abs/10.1145/3378679.3394533&#41;" *Proceedings of the Third ACM International Workshop on Edge Systems, Analytics and Networking*. 2020.)

[//]: # (  - Yang, Jungang, et al. "[Matrix Gaussian Mechanisms for Differentially-Private Learning.]&#40;https://ieeexplore.ieee.org/document/9475590&#41;" *IEEE Transactions on Mobile Computing* &#40;2021&#41;.)

[//]: # (  - Sun, Lichao, Jianwei Qian, and Xun Chen. "[Ldp-fl: Practical private aggregation in federated learning with local differential privacy.]&#40;https://www.ijcai.org/proceedings/2021/217&#41;" *arXiv preprint arXiv:2007.15789* &#40;2020&#41;.)

[//]: # (  - Liu, Ruixuan, et al. "[Fedsel: Federated sgd under local differential privacy with top-k dimension selection.]&#40;https://link.springer.com/chapter/10.1007/978-3-030-59410-7_33&#41;" *International Conference on Database Systems for Advanced Applications*. Springer, Cham, 2020.  )

[//]: # ()
[//]: # (## Remark)

[//]: # ()
[//]: # (The new version uses [Opacus]&#40;https://opacus.ai/&#41; for **Per Sample Gradient Clip**, which limits the norm of the gradient calculated by each sample.)

[//]: # ()
[//]: # (This code sets the number of local training rounds to 1, and the batch size is the local data set size of the client. )

[//]: # (Since the training of the Opacus library will save the gradient of all samples, the gpu memory usage is very large during training.)

[//]: # (This problem can be solved by specifying **--serial** and **--serial_bs** parameters. )

[//]: # ()
[//]: # (These two parameters will physically specify a virtual batch size, and the corresponding training time will be longer, but logically will not affect the training and the addition of DP noise. The main reason for this is to not violate the theory of DP noise addition.)

[//]: # ()
[//]: # (The Dev branch is still being improved, and new DPFL algorithms including MA, F-DP, and Shuffle are implemented in it. Interested friends are welcome to give valuable advice!)

[//]: # ()
[//]: # (## Citation)

[//]: # ()
[//]: # (Consider citing following papers:)

[//]: # ()
[//]: # ([1] W. Yang et al., "Gain Without Pain: Offsetting DP-Injected Noises Stealthily in Cross-Device Federated Learning," in IEEE Internet of Things Journal, vol. 9, no. 22, pp. 22147-22157, 15 Nov.15, 2022, doi: 10.1109/JIOT.2021.3102030.)

[//]: # ()
[//]: # ([2] M. Hu et al., "AutoFL: A Bayesian Game Approach for Autonomous Client Participation in Federated Edge Learning," in IEEE Transactions on Mobile Computing, doi: 10.1109/TMC.2022.3227014.)

[//]: # ()
[//]: # ([3] Y. Zhou et al., "Optimizing the Numbers of Queries and Replies in Convex Federated Learning with Differential Privacy," in IEEE Transactions on Dependable and Secure Computing, doi: 10.1109/TDSC.2023.3234599.)

[//]: # ()
[//]: # ([4] Y. Zhou, et al.,"Exploring the Practicality of Differentially Private Federated Learning: A Local Iteration Tuning Approach" in IEEE Transactions on Dependable and Secure Computing, doi: 10.1109/TDSC.2023.3325889.)

[//]: # ()
[//]: # ([5] Y. Yang, M. Hu, Y. Zhou, X. Liu and D. Wu, "CSRA: Robust Incentive Mechanism Design for Differentially Private Federated Learning," in IEEE Transactions on Information Forensics and Security, doi: 10.1109/TIFS.2023.3329441.)
