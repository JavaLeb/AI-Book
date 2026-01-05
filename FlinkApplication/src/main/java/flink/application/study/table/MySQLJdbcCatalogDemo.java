package flink.application.study.table;

import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.jdbc.core.database.catalog.factory.JdbcCatalogFactoryOptions;
import org.apache.flink.table.api.TableEnvironment;
import org.apache.flink.table.catalog.CatalogDescriptor;
import org.apache.flink.table.catalog.CommonCatalogOptions;


public class MySQLJdbcCatalogDemo {
    public static void main(String[] args) {
        TableEnvironment tEnv = TableAPIUtils.buildTableEnvironment();

        String catalogName = "custom_catalog"; // 目录.
        String defaultDatabase = "mysql"; // 数据库必须存在.

        Configuration config = new Configuration();
        config.set(CommonCatalogOptions.CATALOG_TYPE, JdbcCatalogFactoryOptions.IDENTIFIER);
        config.set(JdbcCatalogFactoryOptions.BASE_URL, "jdbc:mysql://localhost:3306");
        config.set(JdbcCatalogFactoryOptions.USERNAME, "root");
        config.set(JdbcCatalogFactoryOptions.PASSWORD, "root");
        config.set(JdbcCatalogFactoryOptions.DEFAULT_DATABASE, defaultDatabase);

        CatalogDescriptor catalogDescriptor = CatalogDescriptor.of(catalogName, config);
        String logicCatalogName = "logic_custom_catalog";
        tEnv.createCatalog(logicCatalogName, catalogDescriptor);

        tEnv.useCatalog(logicCatalogName);
        // 必须使用 tableEnv.executeSql("show databases")查询出的数据名称.
        // 报错: A database with name [mysql] does not exist in the catalog: [logic_custom_catalog].
        tEnv.useDatabase("dbgpt");

        System.out.println("currentCatalog=" + tEnv.getCurrentCatalog() +
                ",currentDatabase=" + tEnv.getCurrentDatabase());

        tEnv.executeSql("show catalogs").print();
        // 内部对MySQL内置数据库做了过滤, 最终结果不包含mysql等内置数据库.
        tEnv.executeSql("show databases").print();

    }
}
