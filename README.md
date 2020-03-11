# Bayesian-Learning-and-Optimization-Master-Thesis

*This repository consists of experiments for my Master of Statistics thesis at KU Leuven.*

*It is organized in terms of folders that vary chapter-by-chapter.*


**Chapter “Bayesian learning of neural networks”**:

*Exp. 1: A motivating example of Bayesian uncertainty using a neural network for function approximation*\
Goal:  Casting dropout at test-time [Gal & Gahrahmani, 2016] allows for quantification of epistemic and aleatoric uncertainties of a (Bayesian) NN model, and these two forms of uncertainty are nicely visually separated (and motivated) in the context of model selection and data augmentation.

*Exp. 2: A Bayesian neural network classifier that relies on uncertainty when making predictions*\
Goal:  To create, explain, and understand how a Bayesian NN fundamentally differs from a traditional one. To see how the Bayesian formulation allows the NN to refuse making predictions when it feels uncertain at test time, and to see what happens to the predictive accuracy when the decision threshold is varied (the higher, the more strict- the NN will discard more images that it deems too imperfect). The number of samples drawn from the posterior at test-time is also varied. 


**Chapter “Bayesian deep learning”**:

*Exp. 1: A Bayesian convolutional neural network for classifying*\
Goal:  Applying methods like variational inference which are meant to provide tractable alternatives to MacKay and Neal’s original MCMC (for ex.) suggestions.

*Exp. 2: Practical uncertainty modeling in medical imaging*\
Goal:  A common problem in medical imaging, imbalanced datasets create scenarios in which a learner (such as a CNN) is overwhelmed with one class, typically the so-called *control*. As such, its performance at test-time is greatly affected by the scarcity of pathology images, the *treatment*. The goal is to use a pneumonia chest x-ray imaging dataset and define ‘imbalance schemes’ governed by a parameter *p* which denotes the prevalence of the treatment to the control, and observe the effect of such imbalances on the model’s typical metrics like accuracy and f1-score (more appropriate in medical imaging) alongside uncertainty measures as suggested by Gal and Ghahramani (2016).


**Chapter “Bayesian optimization”**:

*Exp. 1: Bayesian optimization for a noisy function*\
Goal:  To show that Bayesian optimization is a guided search method for derivative-free, black box function optimization in a 1-dimensional case. Aleatoric and epistemic uncertainties are represented and reported, with attention to exploration-exploitation (reinforcement learning) of the domain space, and different parameterizations of the GP surrogate and acquisition function.

*Exp. 2: The importance of the acquisition function and GP parameterization in hyperparameter tuning*\
Goal:  To explore how a deep neural network (a Keras model on MNIST) can be tuned via Bayesian optimization, and check relevance of the acquisition function and GP parameterization in optimizing performance.

*Exp. 3: Bayesian optimization as an informed alternative to random search*\
Goal:  Using the same neural network and dataset from the previous experiment, the goal is to statistically test whether Bayesian optimization is truly informed by treating the loss and accuracy as time series of the number of iterations (an analogous treatment is taken for random search). The hypothesis is that a guided search has some trend, while the random search is completely stationary (and alike to white noise).

