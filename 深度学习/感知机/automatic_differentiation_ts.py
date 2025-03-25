import tensorflow as tf

# 定义输入（常量）
x1 = tf.constant(1.0)
x2 = tf.constant(1.0)

# 定义可训练权重（初始值 w1=0, w2=1）
w1 = tf.Variable(0.0)  # 默认会追踪梯度
w2 = tf.Variable(1.0)

# 使用 GradientTape 记录前向计算
with tf.GradientTape() as tape:
    h = (w1 * x1 + w2 * x2) ** 2
    print('输出值', h)
# 计算梯度（目标 h，源 w1 和 w2）
grads = tape.gradient(h, [w1, w2])

# 输出结果
print(f"梯度=({grads[0]}, {grads[1]})")

