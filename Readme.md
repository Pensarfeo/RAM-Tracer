# RAM - MOD

A toy example of how better connecting networks can significantly improve performance by several times!

The experiments are based on mostly making sure that the classifier network is properly connected to the other parts of the network.

This are the experiments with a list of comments

## 

## feedLocationNetwork - WORKS

This network feeds the classifier network to the location network. The intuition is that the location network encodes some info about the expected class and its already encoded by the classifyer. 

## rewardConfusioneDecrease - WRONG IMPLEMENTATION
This was wrong implementation!!!

## noLowerPartGlipmseNet
**Goal**: to understand if the lower part of the glimpse net has any real effect.

**Result**: it does improve the convergence... but **don't know why!**

## allwaysStartFrom0 - WORKS
**Goal**: remove the initial point uncertainty.

**Result**: it does improve the convergence. As the network can learn a better exploration strategy

## removeLocationNoice - NO BUENO
**Goal**: just like the previous one, may be by removing the noise in the location the network will learn more reliable strategies.

**RESULT**: It does not help the network and slows convergence. My guess is that a bit of randomness helps the network discover unexpected exploration paths.

## fastConvergeLoss  - WRONG IMPLEMENTATION
## fastConvergeLoss-measureOnly - WRONG IMPLEMENTATION
## stableConvergeLoss - WRONG IMPLEMENTATION
## stableConvergeLoss-measureOnly - WRONG IMPLEMENTATION

## customSoftMax - WORKS
**GOAL**: to write a custom soft max to use for fast and stable convergence loss experiments

## feedGuessToState - WORKS
**GOAL**: The LSTM network should also have some knowledge of the current expectation. We feed the classifier's output, together with the glimpse network output.

**RESULT**: It works and speeds up the convergence!

## addPreviewNetwork - WORKS
**GOAL**: Rather than starting from the same point, we allow the network to take a very scaled down glimpse of the whole image to select the starting point

**RESULT**: It works and speeds up the convergence!

## addFastConvergence
**GOAL**: we implement fast and stable convergence.

**RESULT**: It does work. Although it does not seem to particularly improve the convergence.

### Description

The proposed loss function is:

$$
loss = p_i \ln{\frac{\sum_{1}^{j=N}\gamma^j p_{i,j}}{N}}
$$

Where $p_i$ is the usual real class probability, $\gamma^j$ is a number smaller than 1, $N$ is the total number of iterations and $p_{i,j}$ is the probability given by the network for the $i^{th}$ class and the $j^{th}$ iteration.

## addFastAndStableConvergence - NO BUENO

**GOAL**: we implement fast and stable convergence.

**RESULT**: It does not work. Although Fast convergence might be working, it does not seem to particularly improve the convergence. Al the same time stable convergence seems to slow down the convergence instead 

## Stable Convergence description

The proposed loss function is:

$$
loss = p_i \ln{(1 - \frac{\sum_{1}^{j=N}\gamma^{N - j + 1} p_{i,j}}{N})}
$$

Where $p_i$ is the usual real class probability, $\gamma^j$ is a number smaller than 1, $N$ is the total number of iterations and $p_{i,j}$ is the probability given by the network for the $i^{th}$ class and the $j^{th}$ iteration.

### Discussion
It is worth noting that Stable convergence, seems to naturally go down in a network where it is not set as a loss. Thus we assume that, either the network is optimal when doing something funny, like waiting till the end to provide a result, or there is something wrong with this loss function.

### Conclusion
It would be interesting to better explore this interaction!
