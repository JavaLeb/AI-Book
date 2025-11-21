package rdd

import org.apache.spark.util.AccumulatorV2


class VectorAccumulator ( val dim: Int) extends AccumulatorV2[Vector, Vector] {

  /**
   * 定义向量累加器属性, 并初始化为零向量.
   */
  private var accVector: Vector = Vector.createZeroVector(dim)

  /**
   * 判断是否为零向量.
   *
   * @return
   */
  override def isZero: Boolean = {
    accVector.isZero()
  }

  /**
   * 重置累积器向量.
   */
  override def reset(): Unit = {
    accVector.reset()
  }

  override def add(v: Vector): Unit = {
    accVector.add(v)
  }

  override def copy(): AccumulatorV2[Vector, Vector] = {
    this
  }

  override def merge(other: AccumulatorV2[Vector, Vector]): Unit = {
    this.value.add(other.value)
  }

  override def value: Vector = {
    accVector
  }
}
