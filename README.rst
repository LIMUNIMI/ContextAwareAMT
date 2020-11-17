The basic idea is to build a model for feature extraction at the note-level.
One model is trained in the source context. Then, a transfer learning approach
is used to train the feature mapping in the target context.

For now, feature extraction is performed using NMF+FF neural network.
In future, NMF should be replaced by NN-methods, so that we can rely on an
end-to-end method.

To make feature-extraction able to generalize over the input context, it should
be trained over different contexts.
