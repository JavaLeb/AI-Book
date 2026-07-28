package flink.application.study.datastream.datasource.datagenconnector;

import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.connector.source.util.ratelimit.RateLimiterStrategy;
import org.apache.flink.connector.datagen.source.DataGeneratorSource;
import org.apache.flink.connector.datagen.source.GeneratorFunction;
import org.apache.flink.streaming.api.datastream.DataStreamSource;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

public class DataGenSourceDemo {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        env.setParallelism(2);

        GeneratorFunction<Long, String> generatorFunction = index -> "Number: " + index;
        long numberOfRecords = 10;

        DataGeneratorSource<String> source =
                new DataGeneratorSource<>(generatorFunction,
                        numberOfRecords,
                        RateLimiterStrategy.perSecond(100),
                        Types.STRING);
        DataStreamSource<String> stream =
                env.fromSource(source,
                        WatermarkStrategy.noWatermarks(),
                        "Generator Source");

        stream.print("|---->");

        env.execute();
    }
}
