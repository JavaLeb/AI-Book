package structured.streaming

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.trim


object WordCountDemo {
  def main(args: Array[String]): Unit = {
    // 创建切入点.
    val spark = SparkSession
      .builder
      .master("local[*]")
      .appName("StructuredNetworkWordCount")
      .getOrCreate()

    // 设置日志级别.
    spark.sparkContext.setLogLevel("WARN")

    // 从socket源读取数据.
    val rawLines = spark.readStream
      .format("socket")
      .option("host", "localhost")
      .option("port", 9999)
      .load()

    // 打印schema.
    rawLines.printSchema()

    val wordWithTime: DataFrame = rawLines
      .selectExpr("split(value, ',', 2) as parts") // 分成两部分（时间戳+单词串）
      .filter(size(col("parts")) === 2) // 过滤无效行（无逗号）
      .select(
        trim(col("parts").getItem(0)).alias("time_str"), // 提取时间戳并去重
        to_timestamp(col("time_str"), "HH:mm").alias("time"),
        explode(split(trim(col("parts").getItem(1)), "\\s+")).alias("word") // 分割单词并展开
      )
      .filter(col("word") =!= "") // 过滤空单词

    wordWithTime.printSchema() // 新增timestamp: timestamp (nullable = true)

    val windowedCounts: DataFrame = wordWithTime
      .groupBy(
        window(col("time"), "10 minutes", "5 minutes"), // 窗口定义
        col("word")
      )
      .agg(count("*").alias("count")) // 统计每个窗口+单词的数量
    windowedCounts.printSchema()

    val formattedCounts: DataFrame = windowedCounts
      .select(
        date_format(col("window.start"), "HH:mm").alias("window_start"),
        date_format(col("window.end"), "HH:mm").alias("window_end"),
        col("word"),
        col("count")
      ).orderBy(col("window_start")) // 按照window_start排序.
    formattedCounts.printSchema()

    val query = formattedCounts.writeStream
      .outputMode("complete")
      .format("console")
      .option("truncate", "false") // 不截断输出
      .start()

    query.awaitTermination()
  }
}
