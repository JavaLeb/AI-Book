package flink.cdc.study.connector;

import org.apache.flink.cdc.cli.parser.PipelineDefinitionParser;
import org.apache.flink.table.api.TableEnvironment;

public class MySQLConnectorDemo {
    public static void main(String[] args) {
        TableEnvironment tEnv = TableAPIUtils.buildStreamTableEnvironment();
        PipelineDefinitionParser
        String createStat = """
                CREATE TABLE orders (
                    db_name       STRING METADATA FROM 'database_name' VIRTUAL,
                    table_name    STRING METADATA FROM 'table_name' VIRTUAL,
                    operation_ts  TIMESTAMP_LTZ(3) METADATA FROM 'op_ts' VIRTUAL,
                    row_kind      STRING METADATA FROM 'row_kind' VIRTUAL,
                
                    order_id INT,
                    order_date TIMESTAMP(0),
                    customer_name STRING,
                    price DECIMAL(10, 5),
                    product_id INT,
                    order_status BOOLEAN,
                    PRIMARY KEY(order_id) NOT ENFORCED
                    ) WITH (
                    'connector' = 'mysql-cdc',
                    'hostname' = 'localhost',
                    'port' = '3306',
                    'username' = 'flinkcdc',
                    'password' = 'flinkcdc',
                    'database-name' = 'flinkcdc',
                    'table-name' = 'orders'
                );
                """;

        tEnv.executeSql(createStat);

        // 打印源表数据.
        String selectStat = """
                select * from orders
                """;
        tEnv.executeSql(selectStat).print();


    }
}
