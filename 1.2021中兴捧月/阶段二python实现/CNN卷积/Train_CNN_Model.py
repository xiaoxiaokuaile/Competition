# Learner: 王振强
# Learn Time: 2021/5/7 23:13
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D,Dropout
from tensorflow.keras.models import load_model

def oned_cnn_model(step_in, step_out, n_features, X, y, epoch_num, verbose_set):
    model = Sequential()

    model.add(Conv1D(filters=100, kernel_size=14, activation='relu',strides=1, padding='valid',
                     data_format='channels_last',input_shape=(step_in, n_features)))
    model.add(Conv1D(filters=200, kernel_size=7, activation='relu',strides=7, padding='valid',
                     data_format='channels_last',input_shape=(step_in, n_features)))
    model.add(MaxPooling1D(pool_size=3, strides=None, padding='valid',data_format='channels_last'))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(units=14, activation='relu',use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', ))

    model.add(Dense(units=step_out,activation='relu'))

    model.compile(optimizer='adam', loss='mse',metrics=['accuracy'], loss_weights=None,
                  sample_weight_mode=None, weighted_metrics=None,target_tensors=None)

    print('\n', model.summary())

    history = model.fit(X, y, batch_size=32, epochs=epoch_num, verbose=verbose_set)

    #yhat = model.predict(test_X, verbose=0)
    #print('\nyhat:', yhat)

    return model, history


if __name__ == '__main__':
    n_steps_in, n_steps_out = 365, 91
    n_feature = 14
    epoch_num = 300
    verbose_set = 0
    XX = np.load(file='dataset/train_X.npy')  # (23820,365,14)
    YY = np.load(file='dataset/train_Y.npy')  # (23820,91)
    train_X = np.array(XX)
    train_y = np.array(YY)
    print(train_X.shape)
    print(train_y.shape)

    model, history = oned_cnn_model(n_steps_in, n_steps_out, n_feature, train_X, train_y, epoch_num,verbose_set)
    model.save('./dataset/train_model.h5')

    # 加载模型
    # model = load_model('./dataset/train_model.h5')
    # 模型评估
    # model.evaluate(x_test,  y_test, verbose=2)

    print('\ntrain_acc:%s' % np.mean(history.history['accuracy']), '\ntrain_loss:%s' % np.mean(history.history['loss']))
