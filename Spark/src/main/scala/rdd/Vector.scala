package rdd

import java.util

/**
 * 向量.
 *
 * @param data 构造参数.
 */
class Vector private(private var data: Array[Double]) extends Serializable {

  private val serialVersionUID: Long = 1L
  /**
   * 向量维度.
   */
  val dim: Int = data.length

  /**
   * 向量加法,原地修改.
   *
   * @param other 另一个向量.
   */
  def add(other: Vector): Unit = {
    // 判断维度.
    require(data.length == other.data.length, "维度不匹配")
    // zip操作：(v1,v2,...) (w1,w2,...) = ((v1,w1),(v2,w2),...)
    // map操作：(v1+w1,v2+w2,...)
    data = this.data.zip(other.data).map { case (a, b) => a + b }
  }

  /**
   * 重置为零向量.
   */
  def reset(): Unit = {
    util.Arrays.fill(data, 0.0)
  }

  /**
   * 判断是否为零向量.
   *
   * @return
   */
  def isZero(): Boolean = this.data.forall(_ == 0.0)

  /** 创建指定维度的零向量 */
  def createZeroVector(): Vector =
    new Vector(Array.fill(dim)(0.0))

  override def equals(obj: Any): Boolean = {
    // 1. 检查是否是同一引用（优化性能）
    if (this.equals(obj)) return true
    // 2. 检查对象是否为Vector类型
    if (!obj.isInstanceOf[Vector]) return false
    // 3. 转换为Vector类型
    val other = obj.asInstanceOf[Vector]
    // 4. 检查维度是否一致
    if (this.dim != other.dim) return false
    // 5. 逐元素比较（基础版：严格数值相等；若需精度可替换为math.abs(a-b) < epsilon）
    this.data.zip(other.data).forall { case (a, b) => a == b }
  }

  def getDataString: String = {
    data.mkString(",")
  }
}

object Vector {

  /**
   * 创建零向量.
   *
   * @param dim 维度.
   * @return
   */
  def createZeroVector(dim: Int): Vector =
    createVector(Array.fill(dim)(0.0))

  def createVector(data: Array[Double]): Vector = new Vector(data)

}
