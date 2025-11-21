package rdd

import org.apache.spark.{SparkConf, SparkContext}

object InitializingSparkDemo {
  def main(args: Array[String]): Unit = {
    val appName = "InitializingSparkDemo"
    val master = "local[*]"
    val conf = new SparkConf().setAppName(appName).setMaster(master)
    val sc = new SparkContext(conf)
    println("sc=", sc)
    println("conf=", conf)

    // 一些操作.

    // 停止.
    sc.stop()
  }
}
