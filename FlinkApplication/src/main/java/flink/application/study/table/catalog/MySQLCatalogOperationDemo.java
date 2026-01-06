package flink.application.study.table.catalog;

import flink.application.study.table.TableAPIUtils;
import org.apache.commons.collections.CollectionUtils;
import org.apache.flink.table.api.*;
import org.apache.flink.table.catalog.*;

import java.util.*;

public class MySQLCatalogOperationDemo {

    public static void main(String[] args) throws Exception {
        TableEnvironment tEnv = TableAPIUtils.buildStreamTableEnvironment();

        String catalogName = "custom_catalog";
        String logicCustomCatalogName = "logic_custom_catalog";
        MySQLCatalogConnectionDemo.registerCatalog(catalogName, logicCustomCatalogName, tEnv);

        Optional<Catalog> catalogOpt = tEnv.getCatalog(logicCustomCatalogName);

        Catalog catalog;
        if (catalogOpt.isPresent()) {
            catalog = catalogOpt.get();
        } else {
            System.err.println("目录不存在，程序退出");
            return;
        }

        String[] catalogs = tEnv.listCatalogs();
        System.out.println("目录列表catalogs=" + Arrays.toString(catalogs));

        List<String> databases = catalog.listDatabases();
        System.out.println("当前数据库列表databases=" + databases);
        String dbName = "my_db";
        catalog.databaseExists(dbName);
        if (CollectionUtils.isEmpty(databases)) {
            return;
        }
        String database = databases.get(0);
        // get database
        CatalogDatabase catalogDatabase = catalog.getDatabase(database);
        System.out.println("database [" + database + "] comment=" + catalogDatabase.getComment());
        List<String> tables = catalog.listTables(database);
        System.out.println("当前数据库[" + database + "]下表=" + tables);

        if (CollectionUtils.isEmpty(tables)) {
            return;
        }
        String tableName = tables.get(0);
        ObjectPath tablePath = new ObjectPath(database, tableName);
        boolean exists = catalog.tableExists(tablePath);
        System.out.println("表[" + tableName + "]是否存在=" + exists);
        CatalogBaseTable catalogTable = catalog.getTable(tablePath);
        System.out.println("表[" + tableName + "]注释=" + catalogTable.getComment() + ",表类型=" +
                catalogTable.getTableKind() + ",表详细描述=" + catalogTable.getDetailedDescription() +
                ",表Schema=" + catalogTable.getUnresolvedSchema());
    }
}
