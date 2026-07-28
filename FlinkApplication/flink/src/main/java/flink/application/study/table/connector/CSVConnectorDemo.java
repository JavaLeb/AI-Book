package flink.application.study.table.connector;

import flink.application.study.table.TableAPIUtils;
import org.apache.flink.core.fs.Path;
import org.apache.flink.table.api.TableEnvironment;
import scala.Predef;

import java.nio.file.Paths;
import java.text.MessageFormat;

public class CSVConnectorDemo {
    public static void main(String[] args) {
        TableEnvironment tEnv = TableAPIUtils.buildTableEnvironment();

        String createSourceTable = """
                                CREATE TABLE fs_source_table (
                                  user_id STRING,
                                  name STRING,
                                  age INT
                                  ,`file.path` STRING NOT NULL METADATA VIRTUAL
                                  ,`file_name` STRING NOT NULL METADATA FROM 'file.name' VIRTUAL
                                  ,`file_size` STRING NOT NULL METADATA FROM 'file.size' VIRTUAL
                                  ,`file.modification-time` STRING NOT NULL METADATA VIRTUAL
                                ) PARTITIONED BY (age) WITH (
                                  'connector'='filesystem',
                                  'path'='data\\sql\\connector\\filesystem\\csv\\input'
                                  ,'format'='csv'
                //                  ,'source.monitor-interval'='3s'
                                );
                """;

        tEnv.executeSql(createSourceTable);
        System.out.println("源表创建成功！");

        String selectStat = """
                SELECT
                    user_id,
                    name,
                    age
                    ,`file.path`
                    ,`file_name`
                    ,`file_size`
                    ,`file.modification-time`
                FROM fs_source_table;
                """;
        tEnv.executeSql(selectStat).print();

//        String createSinkTable = """
//                CREATE TABLE fs_sink_table (
//                  user_id STRING,
//                  name STRING,
//                  age INT
//                  ,`file.path` STRING NOT NULL
//                  ,`file_name` STRING NOT NULL
//                  ,`file_size` STRING NOT NULL
//                  ,`file.modification-time` STRING NOT NULL
//                ) PARTITIONED BY (age) WITH (
//                  'connector'='filesystem',
//                  'path'='file:///AI-Book/FlinkApplication/data/sql/connector/filesystem/csv/output'
//                  ,'format'='csv'
//                );
//                """;
//        tEnv.executeSql(createSinkTable);
//        System.out.println("目标表创建成功！");
//
//
//
//        String insertStat = """
//                INSERT INTO fs_sink_table
//                SELECT
//                    user_id,
//                    name,
//                    age
//                    ,`file.path`
//                    ,`file_name`
//                    ,`file_size`
//                    ,`file.modification-time`
//                FROM fs_source_table;
//                """;
//        System.out.println("从源表查询数据写入目标表");
//        tEnv.executeSql(insertStat);
    }
}
