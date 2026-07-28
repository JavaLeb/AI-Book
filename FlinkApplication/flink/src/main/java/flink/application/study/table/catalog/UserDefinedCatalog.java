package flink.application.study.table.catalog;

import org.apache.flink.configuration.Configuration;
import org.apache.flink.configuration.ReadableConfig;
import org.apache.flink.connector.jdbc.mysql.database.catalog.MySqlCatalog;
import org.apache.flink.table.api.Schema;
import org.apache.flink.table.api.TableNotExistException;
import org.apache.flink.table.catalog.CatalogBaseTable;
import org.apache.flink.table.catalog.CatalogTable;
import org.apache.flink.table.catalog.ObjectPath;
import org.apache.flink.table.connector.source.DynamicTableSource;
import org.apache.flink.table.factories.DynamicTableSourceFactory;

import java.util.Collections;
import java.util.Map;
import java.util.Optional;

public class UserDefinedCatalog {
    public static void main(String[] args) {

    }
//    public class MyCatalogSupportTimeTravel extends MySqlCatalog {
//
//
//        public MyCatalogSupportTimeTravel(ClassLoader userClassLoader, String catalogName, String defaultDatabase, String username, String pwd, String baseUrl) {
//            super(userClassLoader, catalogName, defaultDatabase, username, pwd, baseUrl);
//        }
//
//        @Override
//        public CatalogBaseTable getTable(ObjectPath tablePath, long timestamp)
//                throws TableNotExistException {
//            // Build a schema corresponding to the specific time point.
//            Schema schema = buildSchema(timestamp);
//            // Set parameters to read data at the corresponding time point.
//            Map<String, String> options = buildOptions(timestamp);
//            // Build CatalogTable
//            CatalogTable catalogTable =
//                    CatalogTable.newBuilder()
//                            .schema(schema)
//                            .comment("")
//                            .partitionKeys(Collections.emptyList())
//                            .options(options)
//                            .snapshot(timestamp)
//                            .build();
//            return catalogTable;
//        }
//    }
//
//    public class MyDynamicTableFactory implements DynamicTableSourceFactory {
//        @Override
//        public DynamicTableSource createDynamicTableSource(Context context) {
//            final ReadableConfig configuration =
//                    Configuration.fromMap(context.getCatalogTable().getOptions());
//
//            // Get snapshot from CatalogTable
//            final Optional<Long> snapshot = context.getCatalogTable().getSnapshot();
//
//            // Build DynamicTableSource using snapshot options.
//            final DynamicTableSource dynamicTableSource = buildDynamicSource(configuration, snapshot);
//
//            return dynamicTableSource;
//        }
//    }
}
