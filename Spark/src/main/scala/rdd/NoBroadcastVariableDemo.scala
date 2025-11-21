package rdd

import org.apache.spark.{SparkConf, SparkContext}

import scala.util.Random

object NoBroadcastVariableDemo {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setAppName("NoBroadcastVariableDemo")
      .setMaster("spark://master:7077")
      .set("spark.driver.host", "192.168.128.1")
      .setJars(Seq("target\\Spark-1.0-SNAPSHOT.jar"))
      .set("spark.eventLog.enabled", "true") // 启用事件日志
      .set("spark.eventLog.dir", "file:///spark-history-log") // 日志存储路径（推荐 HDFS）,需要事先创建好.
    val sc = new SparkContext(conf)

    val num = 2_000_00
    val rawData = 1 to num
    val largeLookupMap = rawData.map(i => (i.toString, s"value_$i"))
      .toMap

    val rdd = sc.parallelize(rawData.map(_ => Random.nextInt(2 * num).toString), 100)

    // 不使用广播变量：每个Task都会携带largeLookupMap的副本
    val resultWithoutBroadcast = rdd.map { key =>
      largeLookupMap.getOrElse(key, "not_found") // largeLookupMap会被序列化到每个Task
    }.count() // 触发Action，执行计算

    println(s"不使用广播的结果：$resultWithoutBroadcast")

    sc.stop()
  }
}
