package flink.application.study.table.connector;

import flink.application.study.table.TableAPIUtils;
import org.apache.flink.table.api.TableEnvironment;

public class PrintConnectorDemo {
    public static void main(String[] args) {
        TableEnvironment tEnv = TableAPIUtils.buildTableEnvironment();
        String createStat = """
                CREATE TABLE Orders (
                    boolean_f    BOOLEAN,
                    bigint_f     BIGINT
                ) WITH (
                  'connector' = 'datagen',  -- 连接器类型.
                  'number-of-rows'='10',    -- 表的总行数.
                  'scan.parallelism'='2',   -- 并行度.
                  'fields.bigint_f.kind'='sequence',    -- 生成器类型.
                  'fields.bigint_f.start'='1',          -- 开始值.
                  'fields.bigint_f.end'='100',          -- 结束值.
                  'fields.boolean_f.null-rate'='0.5'    -- null值比率.
                )
                """;
        tEnv.executeSql(createStat);

        String printCreateStat= """
                CREATE TABLE print_table WITH (
                'connector' = 'print',
                'sink.parallelism'='2',
                'print-identifier'='-->',
                'standard-error'='true'
                ) LIKE Orders (EXCLUDING ALL)
                """;

        tEnv.executeSql(printCreateStat);

        String insertStat = """
                INSERT INTO print_table
                SELECT
                    boolean_f,
                    bigint_f
                FROM Orders;
                """;
        tEnv.executeSql(insertStat);

    }
}
