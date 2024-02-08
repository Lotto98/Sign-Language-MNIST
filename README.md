# Sign-Language-MNIST

Final project for FOUNDATION OF MACHINE LEARNING exam (master's degree)

Create a robust classifier to recognize American Sign Language hand poses.

## Data

28x28 pixels images of hand poses:

- 24 categories: full English alphabet excluding J and Z which require motion.
- 27455 training images:
  * 80% actual training.
  * 20% validation.
- 7172 test images.

## Base architectures trained

- ”LeNet5” (1 architecture): LeNet5 architecture from original paper.
- ”Classifier 2” (12 architectures): CNN with 2 convolutional layers.
- ”Classifier 3” (24 architectures): CNN with 3 convolutional layers.

”Classifier 2” and ”Classifier 3” architectures generated by varying:
  - dropout layer positions;
  - number of neurons in hidden layer.

TOTAL: 37 architectures
