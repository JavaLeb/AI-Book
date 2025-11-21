package rdd

import org.apache.spark.{SparkConf, SparkContext}

object ExternalDatasetsDemo {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("ParallelizedCollectionsDemo").setMaster("local[*]")
    val sc = new SparkContext(conf)
    // 设置日志级别.
//    sc.setLogLevel("WARN")
    val distFile = sc.textFile("data/rdd/data.txt")
    val result = distFile.map(s => s.length).reduce((a, b) => a + b)
    println("result", result)
  }
}
