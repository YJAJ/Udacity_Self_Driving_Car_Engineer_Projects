from keras.models import Model, load_model, Sequential
from keras.layers import Input, Conv2D, Activation, BatchNormalization, Dense, \
LeakyReLU, GlobalAveragePooling2D, Dropout, Flatten, Lambda, add
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.initializers import he_normal
from keras.regularizers import l2
from typing import Collection
import argparse
import os
from data import DataLoader, generate_batch

def dbl(input_data, soutput:int, skernel:int=3, stride:int=1, weight_decay=5e-04):
    """Retruns a set of layers consists of Conv2D+batch_norm+Leaky_relu with alpha 0.1"""
    x = Conv2D(soutput, (skernel, skernel), padding='same', strides=stride, kernel_regularizer=l2(weight_decay))(input_data)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                 beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                                 moving_variance_initializer='ones')(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

def res_unit(input_data, soutput:int):
    """Returns two dbl layers with residual connection -
    first dbl half-sized output and second dbl with the original size of outputs"""
    x = dbl(input_data, soutput//2, 1)
    x = dbl(x, soutput, 3)

    return add([input_data, x])

def res_unit_block(input_data, soutput:int, num_block:int):
    """Returns the specified numbers of residual units e.g. 2 num_blocks returns 2 blocks of residual units"""
    x = res_unit(input_data, soutput)

    for i in range(num_block-1):
        x = res_unit(x, soutput)

    return x

def darknet53(sinput:Collection[int], num_blocks:Collection[int], output_sz:int=32):
    """Returns 53 layers of dbl with residual connections"""

    input_shape = Input(shape=sinput)
    x = Lambda(lambda x: x/255.0-0.5, input_shape=sinput)(input_shape)

    #dbl in 3 channels, out 32 channels
    x = dbl(x, output_sz, 3, 1)

    #first - one dbl one res units - in 32 channels, out 64 channels
    x = dbl(x, output_sz*2, 3, 1)
    x = res_unit_block(x, output_sz*2, num_blocks[0])

    #second - one dbl two res units - in 64 ch, out 128 ch
    x = dbl(x, output_sz*4, 3, 2)
    x = res_unit_block(x, output_sz*4, num_blocks[1])

    #third - one dbl eight res units - in 128 ch, out 256 ch
    x = dbl(x, output_sz*8, 3, 2)
    x = res_unit_block(x, output_sz*8, num_blocks[2])

    #four - one dbl eight res units - in 256 ch, out 512 ch
    x = dbl(x, output_sz*16, 3, 2)
    x = res_unit_block(x, output_sz*16, num_blocks[3])

    #five - one dbl four res units - in 512 ch, out 1024 ch
    x = dbl(x, output_sz*32, 3, 2)
    x = res_unit_block(x, output_sz*32, num_blocks[4])

    x = GlobalAveragePooling2D()(x)
    x = Dense(units=1)(x)

    model = Model(inputs=input_shape, outputs=x)

    return model

def training_pipeline(data, model, learning_rate, filepath, bs, epoch, X_train, X_valid, y_train, y_valid):
    save_best = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True,
                    save_weights_only=False, mode='min', period=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.005, patience=2, verbose=0,
                                   mode='min', baseline=None, restore_best_weights=False)

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))
    step_train = len(X_train)*1//bs+1
    step_valid = len(X_valid)*1//bs+1

    #https://keras.io/models/sequential/
    model.fit_generator(generate_batch(X_train, y_train, bs, data), steps_per_epoch=step_train, epochs=epoch, verbose=1, callbacks=[save_best, early_stopping],
                        validation_data=generate_batch(X_valid, y_valid, bs, data), validation_steps=step_valid,
                        class_weight=None, max_queue_size=10, workers=0, use_multiprocessing=False, shuffle=True, initial_epoch=0)

def main():
    pars = argparse.ArgumentParser(description="Training pipeline for Behavioral Cloning")
    pars.add_argument('-d', help="Directory for a dataset", dest='data_path', type=str, default="data/data")
    pars.add_argument('-f', help="CSV file to read in for a path for images and a list of steering angles", dest='file_name', type=str, default="resampled_log.csv")
    pars.add_argument('-l', help="Learning rate", dest='lr', type=float, default=1e-02)
    pars.add_argument('-e', help="Number of epochs", dest='epoch', type=int, default=2)
    pars.add_argument('-b', help="Batch size", dest='bs', type=int, default=8)

    args = pars.parse_args()

    #load the data and split the data into train and valid set
    data = DataLoader(args.data_path, (160, 320, 3), args.file_name)
    X_train, X_valid, y_train, y_valid = data.split_data()
    #create an instance of Darknet53
    model = darknet53((90, 320, 3), [1, 2, 8, 8, 4])
    #train the models with learning rate decay
    lr = args.lr
    for i in range(args.epoch):
        epoch = i+1
        print(f"Learning rate for Epoch {epoch}: {lr}")
        weight_file = 'model_'+str(i)+'.h5'
        training_pipeline(data, model, lr, os.path.join(args.data_path, weight_file), args.bs, 1, X_train, X_valid, y_train, y_valid)
        #reduce learning rate after one epoch
        lr /= 10

if __name__ == '__main__':
    main()
