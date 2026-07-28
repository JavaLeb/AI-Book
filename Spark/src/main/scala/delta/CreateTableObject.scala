package delta


import DeltaSparkSessionBuilder.spark
import io.delta.tables.DeltaTable

object CreateTableObject {
  def main(args: Array[String]): Unit = {
    // Create table in the metastore
    DeltaTable.createOrReplace(spark)
      .tableName("person")
      .addColumn("id", "INT")
      .addColumn(
        DeltaTable.columnBuilder("name")
          .dataType("STRING")
          .comment("姓名")
          .build())
      .addColumn("age", "INT")
      .execute()

    // Create or replace table with path and add properties
    DeltaTable.createOrReplace(spark)
      .addColumn("id", "INT")
      .addColumn(
        DeltaTable.columnBuilder("name")
          .dataType("STRING")
          .comment("姓名")
          .build())
      .addColumn("age", "INT")
      .property("description", "table with person data")
      .location("/tmp/delta/path_person")
      .execute()
  }
}
