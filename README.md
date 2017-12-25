# Incremental Signal Reconstruction

This project implements An Incremental Algorithm for Signal Reconstruction from Short-Time Fourier
Transform Magnitude

## Background

There is the original [paper](http://web.mit.edu/jvb/www/papers/signalrec_ICSLP06.pdf) published in 2006, where authors presented algorithm for signal reconstruction from short-time Fourier transform magnitude. Relying on the experimental results, this algorithm tend to be far better than [Griffin Lim 1985](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.331.7151&rep=rep1&type=pdf).
Also there is [audio samples](http://web.mit.edu/jvb/www/signalrec/) from original work.

## Getting Started

The implementation is in the sigrecon.py file. Also there are sone notebooks to introduce usage example
and comparison with the Griffin Lim algorithm

### Prerequisites

To run the implementation code you need:
* librosa
* numpy
* scipy

Also you need Anaconda package to run introducing notebooks



## Authors

**Zhuk Ivan**


## License

This project is licensed under the MIT License

## Acknowledgments

* Jake Bouvrie jvb@mit.edu
* Tony Ezzat tonebone@mit.edu
* [librosa](https://github.com/librosa/librosa) McFee, Brian, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto
