package rdd

import org.apache.spark.{SparkConf, SparkContext}

object ElementPrintDemo {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setAppName("ElementPrintDemo")
      .setMaster("spark://master:7077")
    //本地连接集群使用.
    //      .set("spark.driver.host", "192.168.128.1")
    val sc = new SparkContext(conf)
    // 设置日志级别.
    sc.setLogLevel("WARN")

    val data = Array(1, 2, 3, 4, 5)
    val rdd = sc.parallelize(data, 3)

    rdd.mapPartitionsWithIndex { (index, element) =>
      element.map(x => s"Partition $index: $x---")
    }.collect().foreach(println)

    rdd.mapPartitionsWithIndex { (index, element) =>
      element.map(x => s"Partition $index: $x,,,")
    }.foreach(println)

    rdd.mapPartitionsWithIndex { (index, element) =>
      element.map(x => s"Partition $index: $x|||")
    }.take(5).foreach(println)

    rdd.mapPartitionsWithIndex { (index, element) =>
      element.map(x => s"Partition $index: $x---")
    }.collect().foreach(println)

  }
}
