## About

Using the camshift algorithm to first detect and track the hand in a live stream. The idea is to first detect the face, and calculate its histogram, then by setting the probability (backprojection) of the face to zero, forcing the algorithm to detect the hand instead. 

The project then builds a dataset, trains a model, and uses this model to classify the hand gesture used in the live stream.