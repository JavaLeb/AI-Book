package bug;

import org.apache.flink.connector.file.src.FileSourceSplit;
import org.apache.flink.core.fs.Path;
import org.apache.flink.table.api.EnvironmentSettings;
import org.apache.flink.table.api.TableEnvironment;

import java.nio.file.Paths;

public class CSVConnectorBugR {
    public static void main(String[] args) {

        // 源码：   Paths.get(split.path().getPath()).getFileName();
        //   Path path = split.path()

        // 如果是data/input
        // 源码会解析为：file:/D:/AI-Book/FlinkApplication/data/input/user.csv
        Path path = new Path("file:/D:/AI-Book/FlinkApplication/data/input/user.csv");

//        String parent =path.getPath();
//        java.nio.file.Path result = Paths.get(parent);        // 报错.
//        String fileName=result.getFileName().toString();

        // 修正.
        String fileName2 = path.getName();


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
