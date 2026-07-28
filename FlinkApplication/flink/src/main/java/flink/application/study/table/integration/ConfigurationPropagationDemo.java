package flink.application.study.table.integration;

import org.apache.flink.configuration.Configuration;
import org.apache.flink.core.execution.CheckpointingMode;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;

import java.time.ZoneId;

public class ConfigurationPropagationDemo {
    public static void main(String[] args) {

        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        env.setMaxParallelism(256);
        // 设置CheckpointConfig.
        env.getCheckpointConfig().setCheckpointingConsistencyMode(CheckpointingMode.EXACTLY_ONCE);
        // 设置ExecutionConfig.
        env.getConfig().setParallelism(2);
        // 通过Configuration来配置.
        Configuration configuration = new Configuration();
        configuration.setString("execution.checkpointing.interval", "1000");
        env.configure(configuration);

        // 在内部env的ExecutionConfig被设置到CatalogManager中
        StreamTableEnvironment tableEnv = StreamTableEnvironment.create(env);
        // 设置TableConfig, 内部会设置到Configuration中.
        tableEnv.getConfig().setLocalTimeZone(ZoneId.of("Europe/Berlin"));
        System.out.println("TableConfig=" + tableEnv.getConfig().getConfiguration());

    }
}
