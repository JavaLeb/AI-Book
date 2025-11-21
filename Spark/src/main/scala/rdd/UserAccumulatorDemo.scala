package rdd

import org.apache.spark.{SparkConf, SparkContext}

object UserAccumulatorDemo {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
      .setAppName("UserAccumulatorDemo")
      .setMaster("spark://master:7077")
      .set("spark.driver.host", "192.168.128.1")
      .setJars(Seq("target\\Spark-1.0-SNAPSHOT.jar"))
      .set("spark.eventLog.enabled", "true") // 启用事件日志
      .set("spark.eventLog.dir", "file:///spark-history-log") // 日志存储路径（推荐 HDFS）,需要事先创建好.
    val sc = new SparkContext(conf)

    val accum = new VectorAccumulator(3)

    sc.register(accum, "VectorAccumulator")
    val v1 = Vector.createVector(Array(0.0, 0.1, 0.2))
    val v2 = Vector.createVector(Array(0.3, 0.4, 0.5))
    val v3 = Vector.createVector(Array(0.6, 0.7, 0.8))
    sc.parallelize(Array[Vector](v1, v2, v3), 2)
      .foreach(x => accum.add(x))
    println(accum.value.getDataString)
    sc.stop()
  }
}
