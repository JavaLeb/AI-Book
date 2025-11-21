package rdd

import org.apache.spark.{SparkConf, SparkContext}

object ParallelizedCollectionsDemo {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("ParallelizedCollectionsDemo").setMaster("local[*]")
    val sc = new SparkContext(conf)
    // 设置日志级别.
    sc.setLogLevel("WARN")

    println("默认并行度", sc.defaultParallelism)

    val data = Array(1, 2, 3, 4, 5)
    val distData = sc.parallelize(data)

    data.foreach(println) // 最终元素会按顺序打印.

    distData.mapPartitionsWithIndex { (index, element) =>
      element.map(x => s"Partition $index: $x")
    }.collect().foreach(println)

    val distData_2 = sc.parallelize(data,3)

    distData_2.mapPartitionsWithIndex { (index, element) =>
      element.map(x => s"Partition $index: $x")
    }.collect().foreach(println)

  }
}
