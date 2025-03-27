import numpy as np
import keras
import matplotlib.pyplot as plt
import pandas as pd

# 异或问题的输入样本.
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]
# 输入数据转换成数组.
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

keras.utils.set_random_seed(42)

cls = keras.Sequential()  # 顺序模型，由顺序连接的层组成.
cls.add(keras.layers.Input(shape=(2,)))  # 输入层.
cls.add(keras.layers.Dense(units=2, activation='sigmoid', kernel_initializer='glorot_normal'))  # 隐藏层.
cls.add(keras.layers.Dense(units=1, activation='sigmoid', kernel_initializer='glorot_normal'))  # 输出层.

cls.summary()  # 输出神经网络结构.
print(cls.layers)  # 输出神经网络层.

for layer in cls.layers:  # 输出初始权重.
    print(layer.get_weights())

# 编译模型.
cls.compile(loss=keras.losses.binary_crossentropy,
            optimizer=keras.optimizers.Adam(learning_rate=0.1),
            metrics=['accuracy'])

history = cls.fit(X, y, epochs=100, batch_size=2, verbose=1, validation_data=(X, y))  # 训练模型.
loss, accuracy = cls.evaluate(X, y)  # 评估数据集损失和准确率.
print('loss=', round(loss, 2), 'accuracy=', accuracy)

y_prob = cls.predict(X)
y_predict = np.where(y_prob > 0.5, 1, 0)
print(y_predict)
for layer in cls.layers:  # 输出最终权重.
    print(layer.get_weights())

pd.DataFrame(history.history).plot(grid=True, xlabel='epoch')  # 绘制学习曲线.
plt.show()

model_file = 'keras_xor_model.h5'
cls.save(model_file)  # 保存模型.
model = keras.models.load_model(model_file)  # 加载模型.
y_prob = model.predict(X)
y_predict = np.where(y_prob > 0.5, 1, 0)
print(y_predict)
