package sql

import org.apache.spark.sql.SparkSession

object SparkSessionTool {

  val spark: SparkSession = SparkSession
    .builder()
    .master("local[*]")
    .appName("SparkSessionDemo")
    .config("spark.some.config.option", "some-value")
    .config("spark.log.level", "WARN")
    .getOrCreate()


}
