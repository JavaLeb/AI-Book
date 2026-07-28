package flink.application.study.sql;

import flink.application.study.table.TableAPIUtils;
import org.apache.flink.table.api.TableEnvironment;

public class CreateCatalogDemo {
    public static void main(String[] args) {
        TableEnvironment tEnv = TableAPIUtils.buildTableEnvironment();
        String statement = "CREATE CATALOG IF NOT EXISTS study_catalog\n" +
                "  COMMENT '我的第一个目录'\n" +
                "  WITH ('type'='generic_in_memory');";
        tEnv.executeSql(statement);
        tEnv.executeSql("SHOW CATALOGS").print();
        // 输出: currentCatalog=study_catalog,currentDatabase=default
        tEnv.useCatalog("study_catalog");
        TableAPIUtils.printCurrent(tEnv);

        // 创建数据库.
        String dbCreateStatement = "CREATE DATABASE IF NOT EXISTS study_catalog.study_db\n" +
                "  COMMENT '我的第一个数据库'\n" +
                "  WITH ('author'='jack');";
        tEnv.executeSql(dbCreateStatement);
        tEnv.executeSql("SHOW DATABASES").print();
        tEnv.useDatabase("study_db");


    }
}
