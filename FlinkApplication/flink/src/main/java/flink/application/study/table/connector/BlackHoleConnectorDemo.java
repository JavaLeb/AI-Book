package flink.application.study.table.connector;

import flink.application.study.table.TableAPIUtils;
import org.apache.flink.table.api.TableEnvironment;

public class BlackHoleConnectorDemo {
    public static void main(String[] args) {
        TableEnvironment tEnv = TableAPIUtils.buildTableEnvironment();
        String createStat = """
                CREATE TABLE Orders (
                    boolean_f    BOOLEAN,
                    bigint_f     BIGINT
                ) WITH (
                  'connector' = 'datagen',  -- 连接器类型.
                  'scan.parallelism'='2',   -- 并行度.
                  'fields.bigint_f.kind'='random',    -- 生成器类型.
                  'fields.bigint_f.min'='1',          -- 开始值.
                  'fields.bigint_f.max'='100',          -- 结束值.
                  'fields.boolean_f.null-rate'='0.5'    -- null值比率.
                )
                """;
        tEnv.executeSql(createStat);

        String printCreateStat= """
                CREATE TABLE blackhole_table WITH (
                'connector' = 'blackhole'
                ) LIKE Orders (EXCLUDING ALL)
                """;

        tEnv.executeSql(printCreateStat);

        String insertStat = """
                INSERT INTO blackhole_table
                SELECT
                    *
                FROM Orders;
                """;
        tEnv.executeSql(insertStat);

    }
}
