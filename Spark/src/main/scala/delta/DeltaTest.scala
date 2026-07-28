package delta

import org.apache.spark.sql.SparkSession

object DeltaTest {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession
      .builder()
      .master("local[*]")
      .appName("DeltaTest")
      .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
      .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
      .config("spark.log.level", "WARN")
      .getOrCreate()

    val df = spark.read.format("delta").load("/tmp/delta-table")
    df.show()

    spark.stop()
  }
}
