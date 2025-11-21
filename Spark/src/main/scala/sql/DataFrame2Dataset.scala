package sql
import SparkSessionTool.spark
import org.apache.spark.sql.DataFrame


object DataFrame2Dataset {
  def main(args: Array[String]): Unit = {

    // DataFrames can be converted to a Dataset by providing a class. Mapping will be done by name
    val path = "examples/src/main/resources/people.json"
    val df: DataFrame = spark.read.json(path)

    import spark.implicits._
    val peopleDS = df.as[Person]

    peopleDS.show()
    // +----+-------+
    // | age|   name|
    // +----+-------+
    // |null|Michael|
    // |  30|   Andy|
    // |  19| Justin|
    // +----+-------+
  }
}
