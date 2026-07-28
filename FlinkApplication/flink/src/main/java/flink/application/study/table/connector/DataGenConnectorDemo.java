package flink.application.study.table.connector;

import flink.application.study.table.TableAPIUtils;
import org.apache.flink.table.api.TableEnvironment;

public class DataGenConnectorDemo {
    public static void main(String[] args) {
        TableEnvironment tEnv = TableAPIUtils.buildTableEnvironment();
        String createStat = """
                CREATE TABLE Orders (
                    boolean_f    BOOLEAN,
                    bigint_f     BIGINT,
                    decimal_f    DECIMAL(32,2),
                    date_f       DATE,
                    time_f       TIME,
                    timestamp_f  TIMESTAMP(3),
                    char_f       CHAR(3),
                    varchar_f    VARCHAR(5),
                    string_f     STRING,
                    row_f        ROW<first_name STRING, last_name STRING>,
                    array_int_f  Array<INT>,
                    map_f        Map<INT, STRING>,
                    multiset_f   MULTISET<INT>
                ) WITH (
                  'connector' = 'datagen',  -- 连接器类型.
                  'rows-per-second'='3',    -- 每秒生成数据行数.
                  'number-of-rows'='10',    -- 表的总行数.
                  'scan.parallelism'='2',   -- 并行度.
                  
                  'fields.decimal_f.kind'='random',     -- 字段生成器类型.
                  'fields.decimal_f.min'='1',           -- 最小值.
                  'fields.decimal_f.max'='100',         -- 最大值.
                  
                  'fields.bigint_f.kind'='sequence',    -- 生成器类型.
                  'fields.bigint_f.start'='1',          -- 开始值.
                  'fields.bigint_f.end'='100',          -- 结束值.
                  
                  'fields.timestamp_f.max-past'='100',  -- 时间戳最大过去.
                  
                  'fields.row_f.first_name.length'='2',
                  'fields.row_f.last_name.length'='3',
                  
                  'fields.array_int_f.length'='1',      -- 长度.
                  'fields.array_int_f.element.min'='1',
                  'fields.array_int_f.element.max'='100',
                  
                  'fields.map_f.length'='1',
                  'fields.map_f.key.max'='300',
                  'fields.map_f.key.min'='200',
                  'fields.map_f.value.length'='3',
                  
                  'fields.multiset_f.length'='1',    
                  'fields.multiset_f.element.max'='30',
                  'fields.multiset_f.element.min'='1',
                  
                  'fields.string_f.var-len'='true',     -- 是否生成可变长数据.
                  'fields.string_f.length'='4',         -- 字段的长度.
                  
                  'fields.boolean_f.null-rate'='0.5'    -- null值比率.
                )
                """;
        tEnv.executeSql(createStat);

        String selectStat = """
                SELECT
                    *
                FROM Orders;
                """;
        tEnv.executeSql(selectStat).print();

    }
}
