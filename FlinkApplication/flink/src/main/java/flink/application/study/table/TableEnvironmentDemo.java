package flink.application.study.table;

import org.apache.flink.table.api.TableEnvironment;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;

public class TableEnvironmentDemo {
    public static void main(String[] args) {
        TableEnvironment tEnv = TableAPIUtils.buildTableEnvironment();
        StreamTableEnvironment stEnv = TableAPIUtils.buildStreamTableEnvironment();
    }
}
