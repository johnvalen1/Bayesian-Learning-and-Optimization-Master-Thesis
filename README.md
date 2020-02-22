# Bayesian-Learning-and-Optimization-Master-Thesis

This repository consists of experiments for my Master of Statistics thesis at KU Leuven.

It is organized in terms of folders that vary chapter-by-chapter.


Chapter “Bayesian learning of neural networks”:
Exp. 1: A Bayesian neural network for (classification/regression)
Goal: To create, explain, and understand how a Bayesian NN fundamentally differs from a traditional one, and explore its fundamentals on a dataset of choice.

Exp. 2: A motivating example of Bayesian uncertainty using a neural network for function approximation
Goal: Casting dropout at test-time [Gal & Gahrahmani, 2016] allows for quantification of epistemic and aleatoric uncertainties of a (Bayesian) NN model, and these two forms of uncertainty are nicely visually separated (and motivated) in the context of model selection and data augmentation.


Chapter “Bayesian deep learning”:
Exp. 1: A Bayesian convolutional neural network for classifying 
Goal: Applying methods like variational inference which are meant to provide tractable alternatives to MacKay and Neal’s original MCMC (for ex.) suggestions.

Exp. 2: Bayesian uncertainty modeling in medical imaging
Goal: To segment images and identify areas where aleatoric uncertainty dominates, versus areas of the image where epistemic uncertainty dominate (like in Kendall & Gal’s “What uncertainties do we need in Bayesian deep learning for computer vision?”). This uses a medical imaging dataset consisting of pneumonia chest x-rays.


Chapter “Bayesian optimization”:
Exp. 1: Bayesian optimization for a noisy function
Goal: To show that Bayesian optimization is a guided search method for derivative-free, black box function optimization in a 1-dimensional case. Aleatoric and epistemic uncertainties are represented and reported, with attention to exploration-exploitation (reinforcement learning) of the domain space, and different parameterizations of the GP surrogate and acquisition function.

Exp. 2: Automated hyperparameter tuning for a neural network for a regression problem
Goal: To show how Bayesian optimization can extend to a guided hyperparameter search problem, where the objective function can be the cost function (in regression, RMSE) and the inputs hyperparameters of the model. A multilayer perceptron model is built and used on a diabetes dataset (10 real-valued features, 1 integer-valued target) and is tuned via random search and Bayesian optimization.

Exp. 3: Bayesian hyperparameter optimization for digit recognition
Goal: To explore how Bayesian deep learning models (CNNs) can be tuned via Bayesian optimization, and monitor performance.

