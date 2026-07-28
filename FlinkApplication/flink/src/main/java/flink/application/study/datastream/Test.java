package flink.application.study.datastream;

import org.apache.flink.api.common.ExecutionConfig;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;


public class Test {
    public static void main(String[] args) throws Exception{
            // 1. 创建执行环境并设置并行度
            StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
            env.setParallelism(2);  // 全局并行度设置为2

            // 2. 从Socket读取数据
            DataStream<String> text = env.socketTextStream("localhost", 9999, "\n");

           text.map(new RichMapFunction<String, Object>() {
               @Override
               public Object map(String value) throws Exception {
                   getRuntimeContext();
                   String[] split = value.split(",");
                   return new Word(split[0], 1, Long.valueOf(split[1]));
               }
           });


            // 5. 启动作业
            env.execute("Socket Word Count with Partition Tracking");
        }

}
