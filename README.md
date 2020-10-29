# Easy21
My solutions to [David Silver's Easy21 assignment](https://www.davidsilver.uk/teaching/).

Code and discussion answers are not guaranteed to be correct.

## Monte Carlo Control

![MC Control Vstar](./imgs/1-mc.png)

## TD Learning

TD backup with time-varying alpha and epsilon (see assignment)
![TD MSE Lambda](./imgs/2-mse-lambda.png)
![TD MSE Episodes](./imgs/2-mse-ep.png)

## Linear Function Approximation

TD backup with constant:
epsilon = 0.05,
alpha = 0.01
![LFA MSE Lambda](./imgs/3-mse-lambda.png)
![LFA MSE Episodes](./imgs/3-mse-ep.png)

## Discussion

_What are the pros and cons of bootstrapping in Easy21?_

Bootstrapping allows policy iteration to be done on every time step within a trajectory by using value estimates of 
future states as opposed to the true result from reaching the terminal state. Bootstrapping also avoids propagating 
large terminal rewards to episode state values which cause an initially high mean squared error (see TD learning image 2). The main downside to 
bootstrapping is using a biased estimate of value, although it will still converge to true value in the limit.

_Would you expect bootstrapping to help more in blackjack or Easy21? Why?_

As cards can have negative values in Easy21 unlike in regular blackjack, episodes can be longer in length.
Therefore the benefit of bootstrapping is greater when applied to Easy21, as not bootstrapping allows the reward of 
long episodes to be propagated back to all states within the episode, despite each state being only loosely connected.