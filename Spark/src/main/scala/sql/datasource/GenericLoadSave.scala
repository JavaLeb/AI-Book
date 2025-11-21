package sql.datasource

import org.apache.spark.sql.SaveMode

object GenericLoadSave {
  def main(args: Array[String]): Unit = {

    import sql.SparkSessionTool.spark
    // 读取parquet文件.
    val usersDF = spark.read.load("examples/src/main/resources/users.parquet")
    usersDF.printSchema()
    usersDF.show()

    // 将DataFrame保存为parquet文件.
    val saveFileName = "namesAndFavColors.parquet"
    usersDF.select("name", "favorite_color")
      .write.mode(SaveMode.Overwrite).save(saveFileName)

    val namesAndFavColorsDF = spark.read.load(saveFileName)
    namesAndFavColorsDF.printSchema()
    namesAndFavColorsDF.show()

    val peopleDF1 = spark.read.format("org.apache.spark.sql.json")
      .load("examples/src/main/resources/people.json")
    peopleDF1.select("name", "age").write.format("parquet").save("namesAndAges.parquet")

    // 读取JSON.
    val peopleDF = spark.read.format("json").load("examples/src/main/resources/people.json")
    peopleDF.select("name", "age").write.format("parquet").save("namesAndAges.parquet")

    // 读取CSV.
    val peopleDFCsv = spark.read.format("csv")
      .option("sep", ";")
      .option("inferSchema", "true")
      .option("header", "true")
      .load("examples/src/main/resources/people.csv")

    // 保存为orc.
    usersDF.write.format("orc")
      .option("orc.bloom.filter.columns", "favorite_color")
      .option("orc.dictionary.key.threshold", "1.0")
      .option("orc.column.encoding.direct", "name")
      .save("users_with_options.orc")

    // 保存为parquet.
    usersDF.write.format("parquet")
      .option("parquet.bloom.filter.enabled#favorite_color", "true")
      .option("parquet.bloom.filter.expected.ndv#favorite_color", "1000000")
      .option("parquet.enable.dictionary", "true")
      .option("parquet.page.write-checksum.enabled", "false")
      .save("users_with_options.parquet")

    val sqlParquetDF = spark.sql("SELECT * FROM parquet.`examples/src/main/resources/users.parquet`")
    val sqlCsvDF= spark.sql("SELECT * FROM csv.`examples/src/main/resources/people.csv`")
  }
}
