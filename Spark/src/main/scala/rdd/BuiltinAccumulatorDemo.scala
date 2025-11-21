package rdd

import org.apache.spark.{SparkConf, SparkContext}

object BuiltinAccumulatorDemo {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
      .setAppName("AccumulatorDemo")
      .setMaster("spark://master:7077")
      .set("spark.driver.host", "192.168.128.1")
      .setJars(Seq("target\\Spark-1.0-SNAPSHOT.jar"))
      .set("spark.eventLog.enabled", "true") // 启用事件日志
      .set("spark.eventLog.dir", "file:///spark-history-log") // 日志存储路径（推荐 HDFS）,需要事先创建好.
    val sc = new SparkContext(conf)

    val accum = sc.longAccumulator("My Accumulator")

    sc.parallelize(Array(1, 2, 3, 4), 2).foreach(x => accum.add(x))
    println(accum.value)
    sc.stop()

  }
}
