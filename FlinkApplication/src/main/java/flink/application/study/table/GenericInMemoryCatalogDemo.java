package flink.application.study.table;

import org.apache.flink.configuration.Configuration;
import org.apache.flink.table.api.*;
import org.apache.flink.table.catalog.CatalogDescriptor;
import org.apache.flink.table.catalog.CommonCatalogOptions;
import org.apache.flink.table.catalog.GenericInMemoryCatalogFactoryOptions;

public class GenericInMemoryCatalogDemo {
    public static void main(String[] args) {
        TableEnvironment tEnv = TableAPIUtils.buildTableEnvironment();

        TableAPIUtils.printCurrent(tEnv);

        String catalogName = "custom_catalog";
        String customDatabase = "custom_database";

//        // 旧API.
//        GenericInMemoryCatalog customCatalog = new GenericInMemoryCatalog(catalogName, customDatabase);
//        tEnv.registerCatalog(catalogName, customCatalog);

        // 新API.
        Configuration config = new Configuration();
        // 等价GenericInMemoryCatalog.
        config.set(CommonCatalogOptions.CATALOG_TYPE, GenericInMemoryCatalogFactoryOptions.IDENTIFIER);
        config.set(GenericInMemoryCatalogFactoryOptions.DEFAULT_DATABASE, customDatabase);
        CatalogDescriptor catalogDescriptor = CatalogDescriptor.of(
                catalogName, config);
        // 在TableEnvironment中的逻辑目录名称.
        String logicCustomCatalog = "logic_custom_catalog";
        tEnv.createCatalog(logicCustomCatalog,catalogDescriptor);

        // 切换目录.
        tEnv.useCatalog(logicCustomCatalog);
        tEnv.useDatabase(customDatabase);

        TableAPIUtils.printCurrent(tEnv);

        tEnv.executeSql("show databases").print();

    }


}
