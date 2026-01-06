package flink.application.study.table.catalog;

import flink.application.study.table.TableAPIUtils;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.table.api.TableEnvironment;
import org.apache.flink.table.catalog.CatalogDescriptor;
import org.apache.flink.table.catalog.CommonCatalogOptions;
import org.apache.flink.table.catalog.hive.factories.HiveCatalogFactoryOptions;

public class HiveCatalogConnectionDemo {
    public static void main(String[] args) {
        TableEnvironment tEnv = TableAPIUtils.buildTableEnvironment();
        String catalogName = "custom_catalog";
        String logicCustomCatalogName = "logic_custom_catalog";
        registerCatalog(catalogName, logicCustomCatalogName, tEnv);

        tEnv.executeSql("show catalogs").print();
        tEnv.executeSql("show databases").print();

        TableAPIUtils.printCurrent(tEnv);
    }

    public static void registerCatalog(String catalogName, String logicCatalogName, TableEnvironment tEnv) {
        Configuration config = new Configuration();
        config.set(CommonCatalogOptions.CATALOG_TYPE, HiveCatalogFactoryOptions.IDENTIFIER);
        config.set(HiveCatalogFactoryOptions.HIVE_CONF_DIR, "conf/hive-conf");
        config.set(HiveCatalogFactoryOptions.DEFAULT_DATABASE, "default");
        CatalogDescriptor catalogDescriptor = CatalogDescriptor.of(
                catalogName, config
        );
        tEnv.createCatalog(logicCatalogName, catalogDescriptor);
        tEnv.useCatalog(logicCatalogName);
    }
}
