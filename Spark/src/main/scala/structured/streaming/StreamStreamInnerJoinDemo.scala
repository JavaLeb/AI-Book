package structured.streaming

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.streaming.Trigger

object StreamStreamInnerJoinDemo {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession
      .builder()
      .master("local[*]")
      .appName("StreamStreamInnerJoinDemo")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")


    import spark.implicits._

    // 创建模拟展示流（每秒1行）
    val impressions = spark.readStream
      .format("rate")
      .option("rowsPerSecond", 3)
      .load()
      .select(
        $"value" as "impressionAdId", // 广告ID (0-9)
        $"timestamp" as "impressionTime" // 转换为时间戳类型
      )
      .withColumn("impressionTime",
        expr("impressionTime - (impressionAdId * interval 1 minute)")) // 添加随机时间偏移

    // 创建模拟点击流（每秒1行）
    val clicks = spark.readStream
      .format("rate")
      .option("rowsPerSecond", 3)
      .load()
      .select(
        $"value" as "clickAdId", // 广告ID (0-9)
        $"timestamp" as "clickTime" // 转换为时间戳类型
      )
      .withColumn("clickTime",
        expr("clickTime - (clickAdId * interval 30 second)")) // 添加随机时间偏移

    // 应用事件时间水印
    val impressionsWithWatermark = impressions.withWatermark("impressionTime", "2 hours")
    val clicksWithWatermark = clicks.withWatermark("clickTime", "3 hours")

    // 流-流内连接（基于广告ID和时间窗口）
    val joinedStream = impressionsWithWatermark.join(
      clicksWithWatermark,
      expr(
        """
        clickAdId = impressionAdId AND
        clickTime >= impressionTime AND
        clickTime <= impressionTime + interval 1 hour
      """),
      "inner"
    )

    // 输出结果模式
    val query = joinedStream.select(
        $"impressionAdId".as("adId"),
        $"impressionTime",
        $"clickTime"
      ).writeStream
      .format("console")
      .outputMode("append")
      .trigger(Trigger.ProcessingTime("3 seconds"))
      .option("truncate", "false")
      .start()

    query.awaitTermination()

  }
}
