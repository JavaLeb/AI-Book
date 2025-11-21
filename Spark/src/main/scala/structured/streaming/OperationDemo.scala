package structured.streaming

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

import java.sql.Timestamp

case class DeviceData(id: Long, device: String, deviceType: String, signal: Double,
                      timestamp: Long, processingTime: Timestamp)

object OperationDemo {
  def main(args: Array[String]): Unit = {
    // 创建Spark切入点.
    val spark: SparkSession = SparkSession
      .builder()
      .master("local[*]")
      .appName("OperationDemo")
      .config("spark.sql.session.timeZone", "UTC")
      .getOrCreate()

    // 设置日志级别.
    spark.sparkContext.setLogLevel("WARN")

    // 用 Spark 内置的 `rate` 源生成基础数据（每秒 2 条，模拟设备上报频率）
    val rawData: DataFrame = spark.readStream
      .format("rate") // 使用 rate 源生成带时间戳的递增数据
      .option("rowsPerSecond", 2) // 每秒生成 2 条（模拟 2 个设备的上报频率）
      .option("numPartitions", 1) // 分区数（简单起见设为 1）
      .load()
      .select(
        col("value").as("value"),
        unix_timestamp(col("timestamp")).cast("long").as("timestamp") // 转为 long 类型整数
      )

    // 将基础数据转换为 IoT 设备数据（DeviceData 格式）
    import spark.implicits._

    val df: DataFrame = rawData.select(
      col("value").as("id"),

      // 设备 ID：用递增 ID 取模生成 4 个模拟设备（device_0 ~ device_3）
      concat(lit("device_"), (col("id") % 4) cast "int").as("device"),

      // 设备类型：根据设备 ID 分配（模拟不同类型的传感器）
      when(col("device").endsWith("_0"), "temperature")
        .when(col("device").endsWith("_1"), "humidity")
        .when(col("device").endsWith("_2"), "pressure")
        .otherwise("light").as("deviceType"),

      // 信号值：随机生成 0~100 的数值（模拟传感器读数）
      ( col("id")*5.0).as("signal"), // 保留 2 位小数

      // 事件时间：用当前时间戳（模拟设备上报的时间）
      col("timestamp"),
      current_timestamp().as("processingTime")
    )

    val ds = df.as[DeviceData] // 转换为 Dataset[DeviceData]

    // Select the devices which have signal more than 10
    val highSignalDevicesDF = df.select("id", "device", "signal").where("signal > 10") // using untyped APIs
    val highSignalDevicesDS = ds.filter(_.signal > 10).map(_.device) // using typed APIs

    // Running count of the number of updates for each device type
    val countByDeviceType = df.groupBy("deviceType").count() // using untyped API

    // Running average signal for each device type
    import org.apache.spark.sql.expressions.scalalang.typed
    val avgSignalByType = ds.groupByKey(_.deviceType).agg(typed.avg(_.signal)) // using typed API

    df.createOrReplaceTempView("updates_view")
    val countResult = spark.sql("select count(*) as num from updates_view") // returns another streaming DF

    println("df is streaming=", df.isStreaming)

    val rawDataPrint = rawData.writeStream
      .format("console")
      .option("truncate", "false")
      .outputMode("update") // 更新模式（流式数据持续追加）
      .start()

    val deviceDataPrint = df.writeStream
      .format("console")
      .option("truncate", "false")
      .outputMode("update") // 更新模式（流式数据持续追加）
      .start()

    // --------------------------
    // 4. 输出结果（流式 Sink）
    // --------------------------
    // 输出到控制台（测试用，生产环境可换成 Kafka、文件等）
    val highSignalDevicesQuery = highSignalDevicesDF.writeStream
      .format("console")
      .option("truncate", "false")
      .outputMode("update") // 更新模式（流式数据持续追加）
      .start()

    val countByDeviceTypeQuery = countByDeviceType.writeStream
      .format("console")
      .option("truncate", "false")
      .outputMode("complete") // 完整模式（每次输出全量统计结果）
      .start()

    val countResultQuery = countResult.writeStream
      .format("console")
      .option("truncate", "false")
      .outputMode("complete") // 完整模式（每次输出全量统计结果）
      .start()

    // 等待流式任务终止（测试时阻塞主线程）
    rawDataPrint.awaitTermination(3*1000)
    deviceDataPrint.awaitTermination(3*1000)
    highSignalDevicesQuery.awaitTermination(3*1000)
    countByDeviceTypeQuery.awaitTermination(3*1000)
    countResultQuery.awaitTermination(3*1000)
  }
}
