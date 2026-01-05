package flink.application.study.table;

import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.EnvironmentSettings;
import org.apache.flink.table.api.TableEnvironment;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;

public class TableEnvironmentBuilder {

    public static void main(String[] args) {
        TableEnvironment tEnv = buildTableEnvironment();
        StreamTableEnvironment stEnv = buildStreamTableEnvironment();
    }

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
}
