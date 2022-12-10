# Project

The project utilizes graph convolutional neural networks to predict the price total on a receipt. Several graph convolutions were tested, each resulting in different precision scores. The performance of different graph convolutions can be seen in the table below.

| Convolution | Precision |
| ----------- | ----------- |
| Sage   | 37%       |
| Graph   | 41%        |
| GAT | 29% |
| TAG | 31% |
|Cheby | 24% |

## Running the Script
The results can (nearly) be replicated by running the script as follows in command line:

python train.py

Once done, the script will print the precision scores of the models.
