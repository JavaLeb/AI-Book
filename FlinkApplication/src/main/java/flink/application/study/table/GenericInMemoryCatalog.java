package flink.application.study.table;

import org.apache.flink.configuration.Configuration;
import org.apache.flink.table.api.*;
import org.apache.flink.table.catalog.CatalogDescriptor;

public class GenericInMemoryCatalog {
    public static void main(String[] args) {
        TableEnvironment tEnv = TableEnvironmentBuilder.buildTableEnvironment();

        String currentCatalog = tEnv.getCurrentCatalog();
        String currentDatabase = tEnv.getCurrentDatabase();
        System.out.println("currentCatalog=" + currentCatalog + ",currentDatabase=" + currentDatabase);

        String catalogName = "custom_catalog";
        String customDatabase = "custom_database";
//        // 旧API.
//        GenericInMemoryCatalog customCatalog = new GenericInMemoryCatalog(catalogName, customDatabase);
//        tEnv.registerCatalog(catalogName, customCatalog);

        // 新API.
        Configuration config = new Configuration();
        // 等价GenericInMemoryCatalog.
        config.setString("type", "generic_in_memory");
        config.setString("default-database", customDatabase);
        CatalogDescriptor catalogDescriptor = CatalogDescriptor.of(
                catalogName, config);
        // 在TableEnvironment中的逻辑目录名称.
        String logicCustomCatalog = "logic_custom_catalog";
        tEnv.createCatalog(logicCustomCatalog,catalogDescriptor);

        // 切换目录.
        tEnv.useCatalog(logicCustomCatalog);
        tEnv.useDatabase(customDatabase);

        currentCatalog = tEnv.getCurrentCatalog();
        currentDatabase = tEnv.getCurrentDatabase();
        System.out.println("currentCatalog=" + currentCatalog + ",currentDatabase=" + currentDatabase);

    }
}
