package flink.application.study.datastream.function;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

public class UserDefinedFunction {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        DataStream<String> source = env.fromData("1", "2", "3", "4");

        // 匿名内部类.
        source.map(new MapFunction<String, Integer>() {
            public Integer map(String value) {
                return Integer.parseInt(value);
            }
        });


        // Java Lambdas.
        source.map((MapFunction<String, Integer>) value -> Integer.parseInt(value));
        source.map((MapFunction<String, Integer>) Integer::parseInt);

        // 实现普通函数接口.
        source.map(new MyMapFunction());

        // 实现Rich函数接口.
        source.map(new MyRichMapFunction());

        env.execute();
    }

    static class MyMapFunction implements MapFunction<String, Integer> {
        public Integer map(String value) {
            return Integer.parseInt(value);
        }
    }

    static class MyRichMapFunction extends RichMapFunction<String, Integer> {
        public Integer map(String value) {
            return Integer.parseInt(value);
        }
    }

}
