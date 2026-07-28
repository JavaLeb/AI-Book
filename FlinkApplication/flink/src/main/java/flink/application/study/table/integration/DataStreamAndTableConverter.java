package flink.application.study.table.integration;

import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;

public class DataStreamAndTableConverter {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(2);
        StreamTableEnvironment stEnv = StreamTableEnvironment.create(env);

        DataStream<String> dataStream = env.fromData("Alice", "Bob", "John");

        Table inputTable = stEnv.fromDataStream(dataStream);

        stEnv.createTemporaryView("InputTable", inputTable);
        Table resultTable = stEnv.sqlQuery("SELECT UPPER(f0) FROM InputTable");

        DataStream<Row> resultStream = stEnv.toDataStream(resultTable);

        resultStream.print();

        env.execute();
    }
}
