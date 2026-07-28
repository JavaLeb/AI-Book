package bug;

import org.apache.flink.table.api.EnvironmentSettings;
import org.apache.flink.table.api.TableEnvironment;

public class CSVConnectorBug {
    public static void main(String[] args) {

        // Table Environment.
        EnvironmentSettings settings = EnvironmentSettings
                .newInstance()
                .inStreamingMode()  // 流模式.
                //.inBatchMode()    // 批模式.
                .build();
        TableEnvironment tEnv = TableEnvironment.create(settings);

        String createSourceTable = """
                CREATE TABLE fs_source_table (
                  user_id STRING,
                  name STRING,
                  age INT,
                  file_name STRING NOT NULL METADATA FROM 'file.name' VIRTUAL
                ) WITH (
                  'connector'='filesystem',
                  'path'='data\\input'
                  ,'format'='csv'
                );
                """;

        tEnv.executeSql(createSourceTable);
        System.out.println("源表创建成功！");

        String selectStat = """
                SELECT
                    user_id,
                    name,
                    age,
                    file_name
                FROM fs_source_table;
                """;
        tEnv.executeSql(selectStat).print();
    }
}
