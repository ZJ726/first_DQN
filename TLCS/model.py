import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # kill warning about tensorflow
import tensorflow as tf
import numpy as np
import sys

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

class TrainModel:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim):
        self._input_dim = input_dim   #输入的状态数组的维度
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._model = self._build_model(num_layers, width)   #每一层width=400个神经元


    def _build_model(self, num_layers, width):
        """
        构建全神经网络
        """
        inputs = keras.Input(shape=(self._input_dim,))
        x = layers.Dense(width, activation='relu')(inputs)
        for _ in range(num_layers):
            x = layers.Dense(width, activation='relu')(x)       #输出作为下一层输入
        outputs = layers.Dense(self._output_dim, activation='linear')(x)   #默认激活函数：线性函数

        model = keras.Model(inputs=inputs, outputs=outputs, name='my_model')
        #编译模型(指明损失函数：均方误差，优化器：adam优化)
        model.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=self._learning_rate))  #lr：学习率
        return model
    

    def predict_one(self, state):
        """
        基于单个状态预测动作值
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state)


    def predict_batch(self, states):
        """
        基于一个batch的状态,预测动作值
        """
        return self._model.predict(states)  #keras方法:predict()


    def train_batch(self, states, q_sa):
        """
        用更新的Q值训练神经网络(model.fit(状态，目标Q值即标签，训练终止时的epoch值，不在标准输出流输出日志信息))
        """
        self._model.fit(states, q_sa, epochs=1, verbose=0)   #epoch是在预测q值时用到，epochs=1即为正向反向传播一次，更新一次参数


    def save_model(self, path):
        """
        保存到模型数据HDF5文件
        """
        self._model.save(os.path.join(path, 'trained_model.h5'))
        plot_model(self._model, to_file=os.path.join(path, 'model_structure.png'), show_shapes=True, show_layer_names=True)


    @property
    def input_dim(self):
        return self._input_dim


    @property
    def output_dim(self):
        return self._output_dim


    @property
    def batch_size(self):
        return self._batch_size


class TestModel:
    def __init__(self, input_dim, model_path):
        self._input_dim = input_dim
        self._model = self._load_my_model(model_path)


    def _load_my_model(self, model_folder_path):
        """
        加载存储在指定模型数的文件中的模型
        """
        model_file_path = os.path.join(model_folder_path, 'trained_model.h5')
        
        if os.path.isfile(model_file_path):
            loaded_model = load_model(model_file_path)
            return loaded_model
        else:
            sys.exit("Model number not found")


    def predict_one(self, state):
        """
        从单个状态预测行动值
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state)


    @property
    def input_dim(self):
        return self._input_dim