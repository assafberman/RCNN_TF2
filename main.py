import tensorflow as tf
import keras
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy, KLDivergence
from keras.metrics import CategoricalAccuracy
from sklearn.preprocessing import OneHotEncoder

(cifar100_image_train, cifar100_label_train), (
    cifar100_image_test, cifar100_label_test) = tf.keras.datasets.cifar100.load_data()
encoder = OneHotEncoder(sparse_output=False)
cifar100_label_train_encoded = encoder.fit_transform(cifar100_label_train)
cifar100_label_test_encoded = encoder.transform(cifar100_label_test)

print(cifar100_image_train.shape)
print(cifar100_label_train.shape)
print(cifar100_label_train_encoded.shape)
print(type(cifar100_image_train))
print(type(cifar100_label_train.shape))
print(type(cifar100_label_train_encoded.shape))

NUM_OF_RCLS = 4
NUM_OF_CONVS_IN_RCL = 3

input_layer = tf.keras.layers.Input(shape=(32, 32, 3))
conv1 = tf.keras.layers.Conv2D(filters=192, kernel_size=(5, 5))(input_layer)
output = tf.keras.layers.BatchNormalization()(conv1)
for i in range(NUM_OF_RCLS):
    conv2 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3))(output)
    bn = tf.keras.layers.BatchNormalization()(conv2)
    x = bn
    for j in range(NUM_OF_CONVS_IN_RCL):
        conv2 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), padding='same')(bn)
        rcl2 = tf.keras.layers.Add()([x, conv2])
        bn = tf.keras.layers.BatchNormalization()(rcl2)
    if i == 1:
        output = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(bn)
    if i != NUM_OF_CONVS_IN_RCL - 1:
        output = tf.keras.layers.Dropout(rate=0.8)(bn)
global_maxpool = tf.keras.layers.GlobalMaxPooling2D()(bn)
output_layer = tf.keras.layers.Dense(units=100)(global_maxpool)

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=0.01), loss=KLDivergence(),
              metrics=CategoricalAccuracy())
model.summary()
# tf.keras.utils.plot_model(model, to_file='./model_graph.png', show_shapes=True, show_layer_activations=True)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True,
                                                           min_delta=0.001, verbose=1)
reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_delta=0.001,
                                                          verbose=1)
callbacks = [tensorboard_callback, early_stopping_callback, reduce_lr_callback]
model_fit = model.fit(cifar100_image_train, cifar100_label_train_encoded, epochs=50, batch_size=128,
                      validation_split=0.2, callbacks=callbacks)

model.save('./pre_trained/')
y_pred = model.predict(cifar100_image_test)
m = CategoricalAccuracy()
m.update_state((cifar100_label_test_encoded, y_pred))
print('Accuracy:', m.result()().numpy())
