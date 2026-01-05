package flink.application.study.table;

import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.EnvironmentSettings;
import org.apache.flink.table.api.TableEnvironment;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;

public class TableAPIUtils {

    public static TableEnvironment buildTableEnvironment() {
        // Table Environment.
        EnvironmentSettings settings = EnvironmentSettings
                .newInstance()
                .inStreamingMode()  // 流模式.
                //.inBatchMode()    // 批模式.
                .build();

        return TableEnvironment.create(settings);
    }

    public static StreamTableEnvironment buildStreamTableEnvironment() {
        // Stream Environment.
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        return StreamTableEnvironment.create(env);
    }

    public static void printCurrent(TableEnvironment tEnv){
        String currentCatalog = tEnv.getCurrentCatalog();
        String currentDatabase = tEnv.getCurrentDatabase();
        System.out.println("currentCatalog=" + currentCatalog + ",currentDatabase=" + currentDatabase);
    }
}
