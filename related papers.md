NLU


—————————————NLU—————————————
Attention Is All You Need
provide a self-attention model called transformer

BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
Provide bi-directional training of Language models based on transformer. The pretraining task is next sentence prediction and masked words prediction.

BERT for Joint Intent Classification and Slot Filling
Use BERT for slot filling and intent classification



—————————————arrive new classes—————————————
Learning without forgetting
Require task identifier to add new output layers. New task contains multiple classes. 
Training is performed by new task warm-up and joint training of old and new tasks with shared parameters. Distillation on new data output.

iCaRL- Incremental Classifier and Representation Learning，
Assume new classes arrive sequentially with class identifier. 
Use a single classification layer model for each class to learn data representation, which is further employed to perform nearest-mean classification in the representation space.
use knowledge distillation from pre-updated model on examplars to build a class incremental multi-class KNN classifier.



Meta-learning representations for continual learning, Khurram Javed et al. NIPS2019
representation learning network + prediction network, the representation network is meta learned. Extensions suggest periodically memory and attention.

Learning to Continually Learn, Jeff Clune et al., ECAI 2020
Assume new classes arrive sequentially with class identifier. Assume class number known.
Model: a gate network called nearomodular network that scales the output layer of prediction network in the range (0,1)
Goal: learn the gate mechanism and a good initialization of prediction network
Meta-train-training: 20 sgd updates of a copy of current prediction network parameters on the training set based on a single new class, random initialize output parameters of the new class of prediction net
Meta-train-testing: use the updated prediction network to calculate the loss from the new class and a random sample of past tasks
Meta-train: back propagate the loss to the gate network and the initialization of prediction network and update, 20000 outer-loop updates in experiment

Meta-test-training: employs the gate network and prediction network, freeze all but the last output layer, which is fine-tuned on the meta-test classes with 15 samples. The final weights from meta-training are continually updated without copy
Meta-test-testing: on new samples of the meta-test classes


—————————————arrive new data—————————————
Overcoming catastrophic forgetting in neural networks.  Proceedings of the national academy of sciences, 2017

 Continual learning through synaptic intelligence,  Friedemann Zenke et al. JMLR2017

Gradient episodic memory for continual learning, nips2017
Considers backward transfer as an optimization constraint formulated as the inner product of gradients.


New Metrics and Experimental Paradigms for Continual Learning, CVPR2018
Elaborate the difference between incremental learning and continual learning

Progress & compress: A scalable framework for continual learning. ICML2018

Memory-based parameter adaptation.  Pablo Sprechmann et al. In Proc. of ICLR , 2018

Lifelong learning with dynamically expandable networks.  Jaehong Yoon et al. ICLR 2018

Differentiable plasticity: training plastic neural networks with backpropagation, ICML2018

Efficient lifelong learning with A-GEM. ICLR2019
Replace the multiple constraints of GEM with an average constraint to facilitate efficiency. Use the task descriptor to improve performance. 

Learning to Learn without Forgetting by Maximizing Transfer and Minimizing Interference, Matt Riemer et al. ICLR2019
Provide an objective based on stability and plastic dilemma.

An empirical study of example forgetting during deep neural network learning,  Mariya Toneva et al. ICLR 2019

Characterizing and avoiding negative transfer,  Zirui Wang et al. CVPR 2019


Experience replay for continual learning. Nips2019

Episodic Memory in Lifelong Language Learning, nips2019
data arrives sequentially. Does not require class identifier, require total class number.
Use an experience memory to store the past examples. The examples in the memory are episodically and sparsely selected to update the model.
Solve new data learning problem, assume all classes are known in advance.

Efficient Meta Lifelong-Learning with Limited Memory, emnlp 2020
Extend the above paper by enabling meta-level learning of local adaptation on both training examples and experience rehearsal, such that the data representation is suitable for efficient local adaptation and efficient for remembering past tasks.

Learning Latent Representations to Influence Multi-Agent Interaction, CORL2020
Learning latent intent by encoder-decoder as additional policy output, in line with modular network usage.

Mnemonics Training: Multi-Class Incremental Learning without Forgetting, CVPR2020


2021/3/10
Done: 调研了meta learning在lifelong learning上论文，找到了对已有任务的新数据学习方法，其中有用meta-learnnig改进的论文。
TODO：调研meta learning 改进对由新类别数据集构成的新任务学习的论文，

2021/3/11
Done: 发现meta continual learning 论文，和我们之前的想法类似，这个框架已被用在incremental class场景，可以扩展到incremental sample场景，可补充memory模块。
TODO：讨论meta incremental class learning 方法在incremental sample场景下的问题，mbpa方法在incremental class场景下的问题，讨论通过memory模块结合两种方法的可能性。memory模块参考NTM，memory augmented continual learning，experience replay in RL.


2021/3/12
Done: 阅读论文MER，meta representation for continual learning
TODO: 调研memory augmented continual learning, 与meta learning objective 结合。


2021/3/16
Done: 阅读GEM, A-GEM, MER，三篇论文的思想是如何利用存储在memory的旧任务数据作为优化限制条件和正则化项，但是不关心存取数据样本，采用随机存储（reservior sampling）并在训练时全部读出。
TODO：1. A-GEM实验表明task descriptor可以提高forward transfer的性能，可以考虑采用disentangled learning学习出正交化的task/class 表示。
	     2. learning to continually learn 论文在incremental class learning场景提出的attention output在fully continual learning （once and one-by-one）场景下，由于局部stationary假设不在成立，所以性能会下降。为此我们需要存储旧样本作为replay或者正则化项提高transfer ability。已有的旧样本回忆方法采用single common function for all task，会导致性能次优。为此，我们提出采用attention 学习task-aware predictor function 和 memory replay。
	     3. 看航源提供的meta-entity文档。

2021/3/18
Done：讨论问题定义和memory学习，阅读PIPE论文
TODO：调研meta class incremental learning以及memory的使用

2021/3/23
Done: 阅读INFORMATION THEORETIC META LEARNING WITH GAUSSIAN PROCESSES，考虑information bottleneck训练continual learning。
TODO：下面主要是熟悉当前NLU模型以及在增量场景下的性能；定义IB for continual learning

2021/3/24
Done: 开始阅读Continual Learning for Robotics: Definition, Framework, Learning Strategies, Opportunities and Challenges，推导IB优化目标，是天然的meta learning场景，下一步要考虑怎么在Continual Learning实现
TODO：本周阅读完上面的论文，熟悉BERT代码。

2021/3/25
Done:阅读review论文Continual Learning for Robotics: Definition, Framework, Learning Strategies, Opportunities and Challenges
TODO: 熟悉BERT代码，调研continual NLU论文，数据集，评价指标，熟悉实现

2021/3/29
Done: 弄懂bert代码
TODO:跑通bert代码并理解bert在estimator下的数据流通；调研continual NLU论文，数据集，评价指标，熟悉实现

2021/3/30
Done: 阅读estimator源码帮助理解estimator工作方式
TODO: 跑通代码，调研continual NLU论文，数据集，评价指标，熟悉实现，
exp：ANML+memory
	 continual bert 及其评价指标



https://papers.nips.cc/paper/2015/hash/9232fe81225bcaef853ae32870a2b0fe-Abstract.html

https://arxiv.org/abs/1907.03799 Rehearsal-Free Continual Learning over Small Non-I.I.D. Batches

https://arxiv.org/pdf/1806.06928.pdf Meta Continual Learning

https://www.google.com.hk/search?q=meta+continual+learning&oq=meta+continual++learning&aqs=chrome..69i57j69i60j69i61l2.5992j0j7&sourceid=chrome&ie=UTF-8 meta continual learning

https://www.google.com.hk/search?q=continual+learning+new+class+new+sample&oq=continual+learning+new+class+new+sample&aqs=chrome..69i57j33i160l2.7652j0j7&sourceid=chrome&ie=UTF-8 continual learning new class new sample

https://www.google.com.hk/search?newwindow=1&safe=strict&ei=_6FJYKrMAYHVtAa6s4GgDw&q=lifelong+neural+turing+machine&oq=lifelong+neural+turing+machine&gs_lcp=Cgdnd3Mtd2l6EANQ5933BFiJifgEYLyM-ARoAXAAeACAAYkCiAGCG5IBBDItMTSYAQCgAQGqAQdnd3Mtd2l6wAEB&sclient=gws-wiz&ved=0ahUKEwjqx_bguKfvAhWBKs0KHbpZAPQQ4dUDCA0&uact=5 lifelong neural turing machine












NLU

Towards End-to-End Learning for Dialog State Tracking and Management using Deep Reinforcement Learning, 
use a RL agent to interact with a user simulator and database, the slot filling label and verbal outputs are treated as the actions, DRQN is exploited to integrated the information of multi turns and the action with the maximum Q value is selected as policy output.


Human-Machine Dialogue as a Stochastic Game, SIGDIAL 2015
Dialogue modeling assumption 1) stationary user behavior (environment) 2) task-oriented and user&machine collaborate to achieve the user’s goal
Experiment in a competitive setting

Generative and Discriminative Algorithms for Spoken Language Understanding

A Novel Bi-directional Interrelated Model for Joint Intent Detection and Slot Filling
use a bi-directional RNN to obtain context information, on top of which is a MLP outputting intent2slot and slot2intent relation to help prediction. Training is performed iteratively between two subnetworks to refine the relation vector. The relation information is finally used for the prediction task.


train the model such that the worst performance of the following potential tasks are optimized.
Pretrained model
Artificial new tasks


Joint Slot Filling and Intent Detection via Capsule Neural Networks
Max margin intent classification, using capsule neural network






2021/3/8
implicit mining

What to learn
How to evaluate
How to find bad cases

P16

NLU
Continual 
Bad case mining

Fp fn RL for NLU

Kg bert
Joint bert

Continual learning for intent classification and slot labelling
