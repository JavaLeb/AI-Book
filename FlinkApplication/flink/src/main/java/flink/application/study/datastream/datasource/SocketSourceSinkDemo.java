package flink.application.study.datastream.datasource;


import org.apache.flink.api.common.serialization.SerializationSchema;
import org.apache.flink.streaming.api.datastream.DataStreamSource;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

public class SocketSourceSinkDemo {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(2);

        // 默认delimiter="\n", 默认maxRetry=0.
        DataStreamSource<String> ds = env.socketTextStream("localhost", 9999);
        ds.print("|-ds->");

        env.socketTextStream("localhost", 9999, "\n");

        env.socketTextStream("localhost", 9999, "\n", 3);


        ds.writeToSocket("localhost", 8888, new SerializationSchema<String>() {
            @Override
            public byte[] serialize(String element) {
                String result = element + "\n";
                return result.getBytes();
            }
        });

        env.execute();
    }
}
