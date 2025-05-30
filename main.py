import numpy as np
import pandas as pd
import argparse
import timeit

import theano
import theano.tensor as T
from dbn import DBN

def load_data(dataset, nfold=5):
    data = pd.read_csv(dataset).values

    data = np.random.permutation(data)

    train_set = data[:-int(data.shape[0] / nfold)]
    test_set = data[-int(data.shape[0] / nfold):]

    def shared_dataset(data_xy, borrow=True):
        data_x = data_xy[:,:-1]
        data_y = data_xy[:,-1]
        shared_x = theano.shared(
                np.asarray(data_x, dtype=theano.config.floatX),
                borrow=borrow)
        shared_y = theano.shared(
                np.asarray(data_y, dtype=theano.config.floatX),
                borrow=borrow)

        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (test_set_x, test_set_y)]
    return rval

def build_finetune_functions(self, datasets, batch_size, learning_rate):
    (train_set_x, train_set_y) = datasets[0]
    (test_set_x, test_set_y) = datasets[1]

    # Compute number of minibatches for training, validation and testing.
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_test_batches //= batch_size

    index = T.lscalar('index')  # index to a [mini]batch

    # Compute the gradients with respect to the model parameters.
    gparams = T.grad(self.finetune_cost, self.params)

    # Compute list of fine-tuning updates.
    updates = []
    for param, gparam in zip(self.params, gparams):
        updates.append((param, param - gparam * learning_rate))

    train_fn = theano.function(
        inputs=[index],
        outputs=self.finetune_cost,
        updates=updates,
        givens={
            self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
            self.y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    test_score_i = theano.function(
        [index],
        self.errors,
        givens={
            self.x: test_set_x[index * batch_size: (index + 1) * batch_size],
            self.y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # Create a function that scans the entire test set.
    def test_score():
        return [test_score_i(i) for i in range(n_test_batches)]

    return train_fn, test_score

def main(args):
    hidden_layers_sizes = [args.nunits] * args.nlayers

    print('Load data.')
    datasets = load_data(args.datafile)

    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[1]

    n_in = train_set_x.shape[1].eval()
    n_out = len(set(train_set_y.eval()))

    # Compute number of minibatches for training and testing.
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // args.batch_size

    numpy_rng = np.random.RandomState(123)

    print('Build the model.')
    dbn = DBN(numpy_rng=numpy_rng, n_ins=n_in, hidden_layers_sizes=hidden_layers_sizes, n_outs=n_out)

    start_time = timeit.default_timer()

    print('Pre-training the model.')
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=args.batch_size,
                                                k=args.k)
    # Pre-train layer-wise.
    for i in range(dbn.n_layers):
        # Go through pretraining epochs.
        for epoch in range(args.pretraining_epochs):
            # Go through the training set.
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            lr=args.pretrain_lr))
            print('Pre-training layer {} epoch {} cost'.format(i, epoch), end=' ')
            print(np.mean(c, dtype='float64'))

    end_time = timeit.default_timer()
    print('The pretraining code ran for {:.2f} sec.'.format(end_time - start_time))

    print('Finetune the model.')
    # Get the training and testing function for the model.
    train_fn, test_model = build_finetune_functions(dbn,
        datasets=datasets,
        batch_size=args.batch_size,
        learning_rate=args.finetune_lr
    )

    best_test_loss = np.inf
    test_frequency = n_train_batches
    epoch = 0
    batch_range = np.arange(n_train_batches)

    while epoch < args.training_epochs:
        start_time = timeit.default_timer()
        epoch = epoch + 1
        np.random.shuffle(batch_range)

        for minibatch_index in range(n_train_batches):

            train_fn(batch_range[minibatch_index])
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % test_frequency == 0:
                print('\repoch {}, minibatch {}/{},'.format(epoch, minibatch_index + 1, n_train_batches), end='')

                test_losses = test_model()
                this_test_loss = np.mean(test_losses, dtype='float64')
                print(' test error {:5.3f}%'.format(this_test_loss * 100.), end='')

                # If we got the best test score until now,
                if this_test_loss < best_test_loss:

                    # Save best test score and iteration number.
                    best_test_loss = this_test_loss
                    best_iter = iter


        end_time = timeit.default_timer()
        print(' {:5.3f} sec'.format(end_time - start_time), end='')

        if epoch % 10 == 0:
            print('')

    print('Optimization complete with best test score of {:5.3f}%,'.format(best_test_loss * 100.), end='')
    print(' obtained at iteration {}'.format(best_iter + 1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', default='data/cpi.csv.gz')
    parser.add_argument('--finetune_lr', default=0.1, type=float)
    parser.add_argument('--pretraining_epochs', default=1, type=int)
    parser.add_argument('--pretrain_lr', default=0.01, type=float)
    parser.add_argument('--k', default=1, type=int)
    parser.add_argument('--training_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--nunits', default=2000, type=int)
    parser.add_argument('--nlayers', default=3, type=int)
    args = parser.parse_args()

    np.random.seed(123)
    main(args=args)
