package structured.streaming

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame


object SocketInputSourceDemo {
  def main(args: Array[String]): Unit = {
    // 创建spark切入点.
    val spark: SparkSession = SparkSession
      .builder()
      .master("local[*]")
      .appName("SocketInputSourceDemo")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()

    // 从Socket读取数据.
    val socketDF: DataFrame = spark
      .readStream
      .format("socket")
      .option("host", "localhost")
      .option("port", 9999)
      .load()

    // Returns True for DataFrames that have streaming sources
    println(socketDF.isStreaming)
    // 打印schema.
    socketDF.printSchema
  }
}
