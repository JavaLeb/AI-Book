package flink.application.study.table.catalog;

import flink.application.study.table.TableAPIUtils;
import org.apache.flink.table.api.*;
import org.apache.flink.table.catalog.*;

import java.util.*;

public class HiveCatalogOperationDemo {
    private static Catalog catalog = null;

    public static void main(String[] args) throws Exception {
        TableEnvironment tEnv = TableAPIUtils.buildStreamTableEnvironment();

        String catalogName = "custom_catalog";
        String logicCustomCatalogName = "logic_custom_catalog";
        HiveCatalogConnectionDemo.registerCatalog(catalogName, logicCustomCatalogName, tEnv);

        // 默认方言：DEFAULT.  指的是Flink原生SQL方言.
        SqlDialect sqlDialect = tEnv.getConfig().getSqlDialect();
        System.out.println("sqlDialect=" + sqlDialect);
        // 设置Hive SQL 方言.
        tEnv.getConfig().setSqlDialect(SqlDialect.HIVE);

        Optional<Catalog> catalogOpt = tEnv.getCatalog(logicCustomCatalogName);

        if (catalogOpt.isPresent()) {
            catalog = catalogOpt.get();
        } else {
            System.err.println("目录不存在，程序退出");
            return;
        }

        String[] catalogs = tEnv.listCatalogs();
        System.out.println("catalogs=" + Arrays.toString(catalogs));

        // 数据库操作.
        String dbName = "my_db";
//        operateDatabase(dbName);

        // 表操作.
        String tableName = "my_table";
        operateTable(dbName, tableName);

        // 视图操作.
        String viewName = "myview";


        // 分区操作.
        // drop partition


        // 函数操作.
        String functionName = "my_function";

    }

    public static CatalogDatabase createDatabase(String dbName) throws Exception {
        CatalogDatabase catalogDatabase = new CatalogDatabaseImpl(
                new HashMap<>() {{
                    put("owner", "alice");
                }}, "我的数据库");
        // 会在Hive中创建数据库.
        catalog.createDatabase(dbName, catalogDatabase, true);

        return catalogDatabase;
    }

    public static void operateDatabase(String dbName) throws Exception {
        System.out.println("创建数据库前databases=" + catalog.listDatabases());
        CatalogDatabase catalogDatabase = createDatabase(dbName);
        // check if a database exist
        boolean databaseExists = catalog.databaseExists(dbName);
        System.out.println("databaseExists=" + databaseExists);
        // get database
        catalogDatabase = catalog.getDatabase(dbName);
        System.out.println("catalogDatabase comment=" + catalogDatabase.getComment());

        // alter database，注：不支持修改数据库注释.
        CatalogDatabase newCatalogDatabase = new CatalogDatabaseImpl(
                new HashMap<>() {{
                    put("owner", "jack");
                    put("env", "prod");
                }}, "我的新数据库");
        catalog.alterDatabase(dbName, newCatalogDatabase, false);
        catalogDatabase = catalog.getDatabase(dbName);
        System.out.println("修改后的数据库属性=" + catalogDatabase.getProperties());
        // drop database
        catalog.dropDatabase(dbName, false);
        System.out.println("删除数据库后databases=" + catalog.listDatabases());

    }

    public static void operateTable(String dbName, String tableName) throws Exception {
        // 前面操作已经删除，这里再次创建.
        createDatabase(dbName);

        ObjectPath tablePath = new ObjectPath(dbName, tableName);
        // 表Schema.
        final Schema schema = Schema.newBuilder()
                .column("name", DataTypes.STRING())
                .column("age", DataTypes.INT())
                .build();
        // create table

        CatalogTable catalogTable = CatalogTable.newBuilder().schema(schema).build();
        ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
        ResolvedCatalogTable resolveCatalogTable = CatalogManager.newBuilder().classLoader(classLoader)
                .config(TableConfig.getDefault()).build().resolveCatalogTable(catalogTable);
        catalog.createTable(tablePath, resolveCatalogTable, false);
        // check if a table exist or not
        boolean exists = catalog.tableExists(tablePath);
        System.out.println("表" + tableName + "是否存在=" + exists);
        // get table
        CatalogBaseTable table = catalog.getTable(tablePath);
        System.out.println("table=" + table.getDescription());

//        // drop table
//        catalog.dropTable(tablePath, false);
//        // alter table
//        String newTableName = "my_new_table";
//        final Schema newSchema = Schema.newBuilder()
//                .column("id", DataTypes.INT())
//                .column("name", DataTypes.STRING())
//                .column("age", DataTypes.INT())
//                .build();
//        catalog.alterTable(tablePath, CatalogTable.newBuilder().schema(newSchema).build(), false);
//        // rename table
//        catalog.renameTable(tablePath, newTableName, true);
//
//        // list tables in a database
//        catalog.listTables(dbName);
    }

//    public static void operateView(Catalog catalog,String dbName,String viewName){
//        String newViewName = "my_new_view";
//        ObjectPath viewPath = new ObjectPath(dbName, viewName);
//        // create view
//        catalog.createTable(viewPath, new CatalogViewImpl(...),false);
//        // drop view
//        catalog.dropTable(viewPath, false);
//        // alter view
//        catalog.alterTable(viewPath, new CatalogViewImpl(...),false);
//        // rename view
//        catalog.renameTable(viewPath, newViewName, false);
//        // get view
//        catalog.getTable(viewPath);
//        // check if a view exist or not
//        catalog.tableExists(viewPath);
//        // list views in a database
//        catalog.listViews(dbName);
//    }

//    public static void operatePartition(Catalog catalog,ObjectPath tablePath){
//        // create partition
//        catalog.createPartition(
//                new ObjectPath("mydb", "mytable"),
//                new CatalogPartitionSpec(...),
//        new CatalogPartitionImpl(...),
//        false);
//        catalog.dropPartition(tablePath, new CatalogPartitionSpec(...),false);
//        // alter partition
//        catalog.alterPartition(
//                tablePath,
//                new CatalogPartitionSpec(...),new CatalogPartitionImpl(...),false);
//        // get partition
//        catalog.getPartition(tablePath, new CatalogPartitionSpec(...));
//        // check if a partition exist or not
//        catalog.partitionExists(tablePath, new CatalogPartitionSpec(...));
//        // list partitions of a table
//        catalog.listPartitions(tablePath);
//        // list partitions of a table under a give partition spec
//        catalog.listPartitions(tablePath, new CatalogPartitionSpec(...));
//        // list partitions of a table by expression filter
//        catalog.listPartitionsByFilter(tablePath, Arrays.asList(epr1, ...));
//    }

//    public static void operateFunction(Catalog catalog,String dbName,String functionName) throws Exception{
//        ObjectPath functionPath = new ObjectPath(dbName, functionName);
//        // create function
//        catalog.createFunction(functionPath, new CatalogFunctionImpl(...),false);
//        // drop function
//        catalog.dropFunction(functionPath, false);
//        // alter function
//        catalog.alterFunction(functionPath, new CatalogFunctionImpl(...),false);
//        // get function
//        catalog.getFunction(functionPath);
//        // check if a function exist or not
//        catalog.functionExists(functionPath);
//        // list functions in a database
//        catalog.listFunctions(dbName);
//    }

}
