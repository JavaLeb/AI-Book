package structured.streaming

import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, SparkSession}

object StreamStaticJoinDemo {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession
      .builder()
      .master("local[*]")
      .appName("appName")
      .getOrCreate()
    // 设置日志级别.
    spark.sparkContext.setLogLevel("WARN")

    val staticDf: DataFrame = spark.createDataFrame(Seq(
      ("book", "《Spark Structured Streaming实战》"),
      ("electronics", "智能手机X"),
      ("clothing", "纯棉T恤"),
      ("food", "有机牛奶")
    )).toDF("type", "product_name") // schema: (type: String, product_name: String)

    staticDf.printSchema() // 查看Schema
    staticDf.show() // 打印数据

    val rawDf: DataFrame = spark.readStream
      .format("socket") // 从socket读取流式数据
      .option("host", "localhost") // socket主机
      .option("port", 9999) // socket端口
      .load()


    val streamingDf: DataFrame = rawDf.selectExpr(
        "split(value, ',')[0] AS type",
        "split(value, ',')[1] AS quantity")
      .select(
        col("type"), // 保留类型字段（Join Key）
        col("quantity").cast("int") // 将数量转为整数
      )

    // 内连接.
    val innerJoinedDf = streamingDf.join(staticDf, "type")
    innerJoinedDf.printSchema()

    // 左外连接.
    val leftOuterJoinedDf = streamingDf.join(staticDf, "type", "left_outer")
    leftOuterJoinedDf.printSchema()

    val innerQuery = innerJoinedDf.writeStream
      .outputMode("append") // 左外连接每个流式行生成一个结果，用append模式
      .format("console") // 输出到控制台
      .option("truncate", "false") // 不截断输出
      .start()

    val leftOuterQuery = leftOuterJoinedDf.writeStream
      .outputMode("append") // 左外连接每个流式行生成一个结果，用append模式
      .format("console") // 输出到控制台
      .option("truncate", "false") // 不截断输出
      .start()

    innerQuery.awaitTermination()
    leftOuterQuery.awaitTermination()

  }

}
