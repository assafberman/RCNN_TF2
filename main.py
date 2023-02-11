import datetime

import keras
import tensorflow as tf
from keras.layers import Conv2D, Add, BatchNormalization, Input, Dropout, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
import csv
from tensorboard.plugins.hparams import api as hp

(cifar100_image_train, cifar100_label_train), (
    cifar100_image_test, cifar100_label_test) = tf.keras.datasets.cifar100.load_data()
encoder = OneHotEncoder(sparse=False)
cifar100_label_train_encoded = encoder.fit_transform(cifar100_label_train)
cifar100_label_test_encoded = encoder.transform(cifar100_label_test)
cifar100_image_train = cifar100_image_train / 255.0
cifar100_image_test = cifar100_image_test / 255.0
datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
                             horizontal_flip=True, vertical_flip=False, validation_split=0.2)
datagen.fit(cifar100_image_train)

HP_NUM_RCLS = hp.HParam('num_of_RCLs', hp.Discrete(range(2, 8)))
HP_NUM_CONVS = hp.HParam('num_of_convs', hp.Discrete(range(2, 7)))
METRIC_ACCURACY = hp.Metric('categorical_accuracy', display_name='categorical_accuracy')
METRIC_CROSSENTROPY = hp.Metric('categorical_crossentropy', display_name='categorical_crossentropy')

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(hparams=[HP_NUM_RCLS, HP_NUM_CONVS], metrics=[METRIC_ACCURACY, METRIC_CROSSENTROPY])


def RCL_block(block_input, hparams):
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(block_input)
    stack2 = BatchNormalization()(conv1)

    RCL = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')
    conv2 = RCL(stack2)
    stack3 = Add()([conv1, conv2])
    stack4 = BatchNormalization()(stack3)

    for num_of_convs in range(0, hparams[HP_NUM_CONVS] - 1):
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu',
                       weights=RCL.get_weights())(stack4)
        stack3 = Add()([conv1, conv2])
        stack4 = BatchNormalization()(stack3)

    return stack4


def train_test_model(hparams):
    input_img = Input(shape=(32, 32, 3))
    rconv1 = Conv2D(filters=64, kernel_size=[5, 5], strides=(1, 1), padding='same', activation='relu')(input_img)

    for num_of_rcls in range(hparams[HP_NUM_RCLS]):
        rconv1 = RCL_block(block_input=rconv1, hparams=hparams)
        if num_of_rcls != hparams[HP_NUM_RCLS] - 1:
            rconv1 = Dropout(rate=0.2)(rconv1)
        if num_of_rcls == 1:
            rconv1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(rconv1)

    out = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(rconv1)
    flatten = Flatten()(out)
    prediction = Dense(units=100, activation='softmax')(flatten)

    model = tf.keras.Model(inputs=input_img, outputs=prediction)
    model.compile(optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
                  loss=CategoricalCrossentropy(),
                  metrics=CategoricalAccuracy())
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(f'./logs/RCL_{hparams[HP_NUM_RCLS]}_{hparams[HP_NUM_CONVS]}/',
                                                          histogram_freq=1)
    callbacks = [early_stopping_callback, reduce_lr_callback, tensorboard_callback]
    #callbacks = [tensorboard_callback]
    model.fit(
        datagen.flow(cifar100_image_train, cifar100_label_train_encoded, batch_size=32, subset='training'),
        validation_data=datagen.flow(cifar100_image_train, cifar100_label_train_encoded, batch_size=32,
                                     subset='validation'), epochs=100, batch_size=8, callbacks=callbacks)
    model.save(f'./pre_trained/RCL_{hparams[HP_NUM_RCLS]}_{hparams[HP_NUM_CONVS]}/')
    eval_loss, eval_accuracy = model.evaluate(cifar100_image_test, cifar100_label_test_encoded)
    return eval_loss, eval_accuracy


def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)
        loss, accuracy = train_test_model(hparams)
        tf.summary.scalar('categorical_crossentropy', loss, step=1)
        tf.summary.scalar('categorical_accuracy', accuracy, step=1)


session_num = 0

for num_rcls in HP_NUM_RCLS.domain.values:
    for num_convs in HP_NUM_CONVS.domain.values:
        hparams = {
            HP_NUM_RCLS: num_rcls,
            HP_NUM_CONVS: num_convs
        }
        run_name = f'run-{session_num}'
        print(f'--- Starting trial: {run_name}')
        print({h.name: hparams[h] for h in hparams})
        run(f'logs/hparam_tuning/{run_name}', hparams)
        session_num += 1

# y_pred = model.predict(cifar100_image_test)
# m = CategoricalAccuracy()
# m.update_state(cifar100_label_test_encoded, y_pred)
# print(f'Accuracy: {m.result().numpy()}')