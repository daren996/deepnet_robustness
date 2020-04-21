# For CSC2516 Marker
This repository contains all the code to reproduce our experiments in CSC2516 project final report. Since our code is built on the top of others' code, there are many outstanding bugs we have to handle. So the steps of running our experiments looks weird.

Step 1 \
Follow the below instruction in to install DeepMatcher. Make sure you could run DeepMatcher as normal (by running deepnet_sensitivity/model.py).

Step 2 \
Open the folder of DeepMatcher library in site-package, substitute deepmatcher/runner.py with our deepnet_sensitivity/deepmatcher_substitute_file/runner.py, substitute deepmatcher/models/core.py with our deepnet_sensitivity/deepmatcher_substitute_file/core.py.

Step 3 \
To run our black box attack, please run deepnet_sensitivity/deep_word_bug.py. You could change the global constant PERTURBATION_PERCENTAGE to control the noise percentage. \
It will generate a set of adversarial examples from test set and store in deepnet_sensitivity/sample_data/itunes-amazon/blackbox_attack_data/adversarial_examples.csv \
Then, please edit TO_SPLIT_FOLDER_PATH, TO_SPLIT_F, and FILE_LIST in deepnet_sensitivity/sample_data/splite_dataset.py to split the adversarial_examples.csv into a series of csv file temp0.csv ~ temp20.csv
Finally, please edit the SUB_PATH and FILE_LIST in deepnet_sensitivity/baseline_attack/evaluation_only.py and run it. The final result will be printed out.

Step 4 \
To run our white box attack, please firstly edit the global constant TO_SPLIT_FOLDER_PATH and TO_SPLIT_F in deepnet_sensitivity/sample_data/splite_dataset.py and run it. It will split the test.csv into a series of temp0.csv ~ temp108.csv and store in deepnet_sensitivity/sample_data/itunes-amazon/gradient_attack_data. Each tempi.csv contains one test example. \
Then, please run deepnet_sensitivity/gradient_attack/gradient_attack.py. You could change the global constant NUM_TOKENS to control the number of noise tokens.
Next, please edit the NUM_DELETE in deepnet_sensitivity/sample_data/delete_lastrow.py equal to NUM_TOKENS and run it.
Finally, please edit the SUB_PATH and FILE_LIST in deepnet_sensitivity/baseline_attack/evaluation_only.py and run it. The final result will be printed out.

All of above complicated steps are necessary to avoid the outstanding bug of DeepMatcher. If you have problem on reproducing our experiment, please contact ybzhang@cs.toronto.edu.


# Deep Net Sensitivity
Abstract: Although deep neural networks provide state-of-the-art results for most machine learning tasks, such as natural language processing, recent research has exposed that those are vulnerable to adversarial attacks. The existence of such adversarial examples implies the fragility of deep models. In this project, we will apply adversarial attacks on DeepMatcher, a deep learning (DL) model for entity resolution problem, to explore the methods of attacking a model based on natural language, instead of images, and evaluate its robustness based on the adversarial attack. 

Object: 1) Generate text-based adversarial samples, as the input of DeepMatcher is two text-based tuples, and the output is a binary classification; 2) Define a formulation of robustness; 3) Propose defense methods on text-based models.

Advanced Object: Measure how confident we are that deep nets is robust enough, which is not a general confidence for the predictions, but confidence for defending adversarial attacks

## Deep Models

Focus on entity resolution, entity matching models

### DeepMatcher

[DeepMatcher](https://github.com/anhaidgroup/deepmatcher) is a Python package for performing entity and text matching using deep learning. It provides built-in neural networks and utilities that enable you to train and apply state-of-the-art deep learning models for entity matching in less than 10 lines of code. The models are also easily customizable - the modular design allows any subcomponent to be altered or swapped out for a custom implementation.

Note that DeepMatcher only supports Python versions 3.5 and 3.6.

```shell
pip install deepmatcher
```

Download dataset before running model.py:

```shell script
chmod +x ./scripts/download_data.sh
./scripts/download_data.sh
```

DeepMatcher uses [Pandas](https://pandas.pydata.org/docs/getting_started/10min.html) to manipulate data frames.

The ER is a text-based binary classification problem, with output *Match* or *NoMatch* and a pair of tuples (u,v) as input. In order to evaluate its robustness, we would like to perturb its input pairs and evaluate the accuracy of the perturbed examples. We pre-define a relative small *maximum perturbation* ϵ and assume the truth label does not change after the perturbation. Then, a test set is sufficient for our robustness evaluation.

#### TIPS

1. Most adversarial attack methods are designed for images, while the input of our entity matching is two tuples; thus, how to generate adversarial attacks also needs to be studied.
2. The measurement of the robustness of the model can be i) the size of the perturbation (refer to [DeepFool](https://github.com/daren996/deepnet_sensitivity/blob/master/papers/Adversarial%20Attacks/DeepFool-%20a%20simple%20and%20accurate%20method%20to%20fool%20deep%20neural%20networks.pdf)), or ii) the interpretation of results (refer to [LIME for DeepMatcher](https://github.com/daren996/deepnet_sensitivity/blob/master/papers/Entity%20Resolution/(mandatory)%20Interpreting%20Deep%20Learning%20Models%20for%20Entity%20Resolution%20-%20An%20Experience%20Report%20Using%20LIME.pdf)). The code can be found [here](https://colab.research.google.com/drive/1dR--TdzF7I8qsQPLYn1oc0mvtWnB4ZoY).
3. If computer vision models could use [distillation](https://doi.org/10.1109/SP.2016.41) or ensemble approaches to defense the adversarial attacks, how could we improve the entity matching deep models? 

Three types of entity problems: structure, texture, and dirty. DeepMatcher remove the attribute boundry and make them as text / word sequence. 

Module: Word Embedding, Similarity Representation, and Classifier. 

Embedding types: char, word. Besides, embedding methods can be pre-trained or learned. We will use gradient attack method on char-level embeddings and random generalization perturbation for word-level embeddings. Learned embedding can derive gradience directly. Pre-trained (such as [fastText](https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00051)) need further research. 






## Adversarial Attacks

There are two ways, **black-box attacks** and **white-box attacks**, to general adversarial attacks; or **non-targeted attacks**, to make the classifier misclassify an adversarial example which is crafted by adding some small noise without changing the label, and **target attacks**, aiming to misclassify the example to the target specified label. 

**Clean Example**: If the input sample is a naturally-derived sample, such as a photo from the ImageNet.

**Adversasrial Example**: If an attacker modifies a sample so that the sample can be misclassified.



### Boosting Adversarial Attacks with Momentum

It proposed momentum-based iterative methods (**MI-FGSM**) to boost adversarial attacks, which can effectively fool white-box models as well as black-box models. [Paper](https://github.com/daren996/deepnet_sensitivity/blob/master/papers/Adversarial%20Attacks/Boosting%20Adversarial%20Attacks%20with%20Momentum.pdf)

By integrating the momentum term into the iterative process for attacks, their methods can stabilize update directions and escape from poor local maxima during the iterations, resulting in more transferable adversarial examples. To further improve the success rates for black-box attacks, we apply momentum iterative algorithms to an ensemble of models, and show that the adversarially trained models with a strong defense ability are also vulnerable to our black-box attacks. 

NIPS 2017 上 Ian Goodfellow 牵头组织了 Adversarial Attacks and Defences 竞赛，清华大学的团队在竞赛中的全部三个项目中得到冠军。

#### TIPS

They boost adversarial attacks with **Momentum**. Can we try it on other optimization methods, such as **Adam**?

### FoolBox

[Foolbox](https://foolbox.readthedocs.io/) is a Python toolbox to create adversarial examples that fool neural networks.

### DeepFool

DeepFool is a gradient-ascent based method, focus on multi-class *image* classification. The proposed algorithm operates in a greedy way, which is not guaranteed to converge to the optimal minimal perturbation but practically is a good approximation of it. [[Google Doc]](https://docs.google.com/document/d/1YrxyT2a85qnVKQEWo_Yspt2wva1ETfkGAZ5zq0K5AdY/edit?usp=sharing)

#### Universal adversarial perturbation

Existing image-agnostic universal perturbation vectors can lead to misclassification with high probability across classifiers, but be quasi-imperceptible to humans. 
The perturbation is smaller than a criterion but not optimal.
Solving two minimization problem.
Also, there is a property of *doubly-universal*, which is model-agnostic perturbation.

### HotFlip

It designed a new attack method for text classification (while-box), works on differentiable systems; The method uses gradience with respect to estimate which individual change has the highest loss and "flip" the value to confuse the classifier. [[paper]](https://arxiv.org/abs/1712.06751) 

Three kinds of modification: flip, insert, and delete. We could apply more modification methods along this line. 


## Adversarial Attacks History

2013 - [Intriguing properties of neural networks](https://arxiv.org/abs/1312.6199), the first paper to propose the adversarial attack method against deep networks, **L-BFGS** (known as *Optimizatioin-based Methods*).

2014 - [Explaining and harnessing adversarial example](https://arxiv.org/abs/1412.6572), first proposed **FGSM** (fast gradient sign method, also known as *One-step Gradient-based Approach*), which speeds up L-BFGS but result at low success rate. 

2016 - [Adversarial examples in the physical world](https://arxiv.org/abs/1607.02533), proposed the iterative FGSM (**I-FGSM**, also known as *Basic Iterative Method*, BIM), which apply fast gradient multiple times with a small step. 

2017 - [Towards deep learning models resistant to adversarial attacks](https://arxiv.org/abs/1706.06083), proposed Madry Attacks, which significantly improves BIM by starting at random points within the ε range ball.

2017 - [Adversarial transformation networks: Learning to generate adversarial examples](https://arxiv.org/abs/1703.09387), trained a network to generate adversarial examples.

2017 - [Towards Evaluating the Robustness of Neural Networks](https://doi.org/10.1109/SP.2017.49), designed new loss functions and used Adam to optimize them.

**2017** - **[HotFlip - White-Box Adversarial Examples for Text Classification](https://arxiv.org/abs/1712.06751)**, designed a new attack method for text classification (while-box), works on differentiable systems; The method uses gradience with respect to estimate which individual change has the highest loss and "flip" the value to confuse the classifier. 

2018 - [Generating Natural Language Adversarial Examples](https://arxiv.org/abs/1804.07998), It is an improvement of [HotFlip](https://arxiv.org/abs/1712.06751). The hotflip is white-box and the biggest downside is simple character flipping often leads to meaningless words (e.g., mood -> mooP), the work in this paper uses a black-box population-based optimization algorithm to generate semantically and syntactically similar adversarial examples that fool well-trained sentiment analysis and textual entailment models. [[downsides]](https://github.com/daren996/deepnet_sensitivity/pull/9#issue-385215906)

2019 (AAAI) - [CGMH: Constrained Sentence Generation by Metropolis-Hastings Sampling](https://arxiv.org/abs/1811.10996), I have to say this paper is **awesome**. For RNN, we cannot control the output by given a specific input, so the generated sentence sometimes dosen't make scense. This paper raised CGMH, by using this statistic method, we can easily generate smooth sentence that meets our expect. This method can be appled in out ER problem to generate adversarial samples with better fluency and naturalness, by reading this first will help you to understand the next two ACL paper well.

2019 (ACL) - [Generating Fluent Adversarial Examples for Natural Languages](https://lileicc.github.io/pubs/zhang2019generating.pdf), It is an improvment of _[Generating Natural Language Adversarial Examples](https://arxiv.org/abs/1804.07998)_. Its contribution mainly lies in proposing a more effective method to generate smooth adversarial samples using MHA, based on traditional sampling (Metropolis- hastings (M-H)) methods.

2019 (ACL) - [Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency](https://www.aclweb.org/anthology/P19-1103/), it introduced a new word replacement order determined by both the word saliency and the classification probability, and propose a greedy algorithm called probability weighted word saliency (PWWS) for text adversarial attack. Which improved the [HotFlip](https://arxiv.org/abs/1712.06751) a lot. The code can be found [here](https://github.com/JHL-HUST/PWWS).



All of the above methods can do both target adversarial attacks and non-target adversarial attacks. However, they all need to calculate the gradients of the attacked model, while for a **non-differentiable system**, they suck. For this case, we could train a surrogate model to generate transferable adversarial attack examples for the non-differentiable system.

**Note:** For textual adversarial attack and defense, [here](https://github.com/thunlp/TAADpapers) is a paper list which may be helpful for applications on ER models with text-included dataset.


## Robustness

Naive Definition: Suppose the original DeepMatcher achieves 90% accuracy on a test set, then we apply a best adversarial attack algorithm to generate a set of perturbed examples within maximum ϵ perturbations. The accuracy of DeepMatcher is tested on the set of perturbed examples, where the accuracy on perturbed examples is its robustness. 

After the evaluation above, we could apply some defensive techniques on the DeepMatcher to improve its robustness. Then we get a DeepMatcher-1 which could achieve better accuracy on the set of perturbed examples.

We can continue the iteration of generating perturbed examples, evaluation accuracy on perturbed examples, and then improve the model based on defensive techniques. 

Finally, we may reach two possible endings: 1) The model becomes robust or the robustness of the model can converge, which means the model could achieve high accuracy on the perturbed examples set. 2) The robustness of the model cannot converge, which means after k iterations, a DeepMatcher-k can be robust to (k-1)-th attack, but is fragile to i-th attack, where 1< i < (k-1). 
