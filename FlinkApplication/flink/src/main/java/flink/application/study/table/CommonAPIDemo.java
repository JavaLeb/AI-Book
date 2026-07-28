package flink.application.study.table;

import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.datagen.table.DataGenConnectorOptions;

import org.apache.flink.table.api.*;
import org.apache.flink.table.catalog.CatalogDescriptor;
import org.apache.flink.table.catalog.GenericInMemoryCatalog;

import static org.apache.flink.table.api.Expressions.$;

public class CommonAPIDemo {
    public static void main(String[] args) {
        TableEnvironment tEnv = TableAPIUtils.buildTableEnvironment();
        TableAPIUtils.printCurrent(tEnv);

        tEnv.executeSql("SELECT\n" +
                "  name,\n" +
                "  COUNT(*) AS cnt\n" +
                "FROM\n" +
                "  (VALUES ('Bob'), ('Alice'), ('Greg'), ('Bob')) AS NameTable(name)\n" +
                "GROUP BY name;").print();

//        // Create a source table
//        // 定义 Schema（包含 rowtime 字段）
//        Schema schema = Schema.newBuilder()
//                .column("a", DataTypes.INT())
//                .column("b", DataTypes.STRING())
//                .column("c", DataTypes.INT())
//                .columnByExpression("rowtime", "PROCTIME()") // 生成时间戳
//                .build();
//
//        TableDescriptor tableDescriptor = TableDescriptor.forConnector("datagen")
//                .schema(schema)
//                .option(DataGenConnectorOptions.ROWS_PER_SECOND, 100L)
//                .option(DataGenConnectorOptions.NUMBER_OF_ROWS, 10L)
//                .option("fields.a.kind", "sequence")                 // a 字段随机生成
//                .option("fields.a.start", "1")
//                .option("fields.a.end", "100")
//                .option("fields.b.kind", "sequence")               // b 字段随机字符串
//                .option("fields.b.start", "001")
//                .option("fields.b.end", "100")
//                .option("fields.c.kind", "sequence")             // c 字段序列生成
//                .option("fields.c.start", "1000")                // c 起始值
//                .option("fields.c.end", "9999")                  // c 结束值
//                .build();
//
//        // 注册表,表的path为Orders.
//        tEnv.createTable("Orders", tableDescriptor);
//        // 验证数据（可选）
//        tEnv.executeSql("SELECT * FROM Orders").print();
//        // 类似于SQL查询中的FROM子句。对已注册表执行扫描。
//        Table orders = tEnv.from("Orders");
//
//        Table counts = orders
//                .groupBy($("a"))
//                .select($("a"), $("b").count().as("cnt"));
//
//        // print
//        counts.execute().print();

//        // 注册Table.
//        // full path: default_catalog.default_database.SourceTable
//        tEnv.createTemporaryTable("SourceTemporaryTable", tableDescriptor);
//        tEnv.createTable("SourceTable", tableDescriptor);
//
//        Table table = tEnv.from("SourceTable").select($("id"),$("value"));
//
//        tEnv.createView("SourceView", table);
//        tEnv.createTemporaryView("SourceTemporaryView", table);


//        // Create a sink table (using SQL DDL)
//        tEnv.executeSql("CREATE TEMPORARY TABLE SinkTable WITH ('connector' = 'blackhole') LIKE SourceTable (EXCLUDING OPTIONS) ");
//
//        // Create a Table object from a Table API query
//        Table table1 = tEnv.from("SourceTable");
//
//        // Create a Table object from a SQL query
//        Table table2 = tEnv.sqlQuery("SELECT * FROM SourceTable");
//
//        // Emit a Table API result Table to a TableSink, same for SQL result
//        TableResult tableResult = table1.insertInto("SinkTable").execute();
//
//        tableResult.print();


//        CatalogDescriptor catalogDescriptor= CatalogDescriptor.of(
//                "custom_catalog",
//                new Configuration(),
//                "");
//        tEnv.createCatalog("custom_catalog",catalogDescriptor);

//        // 2. 创建自定义 Catalog（这里用内存 Catalog 举例）
//        String catalogName = "custom_catalog";
//        String defaultDatabase = "custom_database";
//        GenericInMemoryCatalog customCatalog = new GenericInMemoryCatalog(catalogName, defaultDatabase);
//
//        Configuration config = new Configuration();
//        config.setString("type", "generic_in_memory");           // 必须！
//        config.setString("default-database", defaultDatabase);   // 可选，但推荐
//
//        CatalogDescriptor catalogDescriptor = CatalogDescriptor.of(
//                "custom_catalog", config);
//        tEnv.createCatalog("custom_catalog",catalogDescriptor);
//
//        // 3. 将 Catalog 注册到 TableEnvironment
//        tEnv.registerCatalog(catalogName, customCatalog);
//
//
//        tEnv.useCatalog("custom_catalog");
//        tEnv.useDatabase("custom_database");
//
//        final TableDescriptor sourceDescriptor = TableDescriptor.forConnector("datagen")
//                .schema(Schema.newBuilder()
//                        .column("f0", DataTypes.STRING())
//                        .build())
//                .option(DataGenConnectorOptions.ROWS_PER_SECOND, 100L)
//                .build();
//
//        // 从Source创建Permanent Table.
//        tEnv.createTable("SourceTablePath", sourceDescriptor); // 注册表.
//        Table sourceTable = tEnv.from("SourceTablePath"); // 创建表对象.
//
//        // 从Source创建Temporary Table.
//        tEnv.createTemporaryTable("SourceTemporaryTablePath", sourceDescriptor);
//        Table sourceTemporaryTable = tEnv.from("SourceTemporaryTablePath");
//
//        // 从Table创建View.
//        tEnv.createView("SourceViewPath", sourceTable);
//        Table sourceView = tEnv.from("SourceViewPath");
//
//        // 从Table创建Temporary View.
//        tEnv.createTemporaryView("SourceTemporaryViewPath", sourceTable);
//        Table sourceTemporaryView = tEnv.from("SourceTemporaryViewPath");
//
//
//// Create a Table object from a SQL query
//        Table table2 = tEnv.sqlQuery("SELECT * FROM SourceTablePath");
//
//        table2.printSchema();


    }
}
