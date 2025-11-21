package structured.streaming

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.StructType


object FileInputSourceDemo {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession
      .builder()
      .master("local[*]")
      .appName("FileInputSourceDemo")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()

    // Read all the csv files written atomically in a directory
    val userSchema: StructType = new StructType().add("name", "string").add("age", "integer")
    val csvDF: DataFrame = spark
      .readStream
      .option("sep", ";")
      .schema(userSchema) // Specify schema of the csv files
      .csv("data/structuredStreaming/inputSource") // Equivalent to format("csv").load("/path/to/directory")

    // 打印schema.
    csvDF.printSchema
  }
}
