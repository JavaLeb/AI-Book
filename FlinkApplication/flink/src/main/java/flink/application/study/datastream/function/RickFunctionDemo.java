package flink.application.study.datastream.function;

import org.apache.flink.api.common.functions.*;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.streaming.api.TimerService;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.streaming.api.functions.ProcessFunction;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;

public class RickFunctionDemo {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        DataStream<Tuple2<String, String>> stream = env.fromData(new Tuple2<>("", ""));

        DataStream<Tuple2<String, Long>> result = stream
                .keyBy(value -> value.f0)
                .process(new MyProcessFunction());
    }

    public static class MyMapFunction implements MapFunction<Integer, String> {
        @Override
        public String map(Integer value) throws Exception {

            return "map-" + value;
        }
    }

    public static class MyRichMapFunction extends RichMapFunction<Integer, String> {
        @Override
        public String map(Integer value) throws Exception {

            int taskIndex = getRuntimeContext().getTaskInfo().getIndexOfThisSubtask();
            return "taskIndex=" + taskIndex + ",map-" + value;
        }

        @Override
        public void open(OpenContext openContext) throws Exception {
            super.open(openContext);
        }

        @Override
        public void close() throws Exception {
            super.close();
        }
    }

    public static class MyProcessFunction extends ProcessFunction<Tuple2<String, String>, Tuple2<String, Long>> {

        private ValueState<CountWithTimestamp> state;

        @Override
        public void processElement(Tuple2<String, String> value, ProcessFunction<Tuple2<String, String>, Tuple2<String, Long>>.Context ctx, Collector<Tuple2<String, Long>> out) throws Exception {
            int taskIndex = getRuntimeContext().getTaskInfo().getIndexOfThisSubtask();
            TimerService timerService = ctx.timerService();
            ctx.output(new OutputTag<>(""), new Tuple2<>());
            ctx.timestamp();

            // retrieve the current count
            CountWithTimestamp current = state.value();
            if (current == null) {
                current = new CountWithTimestamp();
                current.key = value.f0;
            }

            // update the state's count
            current.count++;

            // set the state's timestamp to the record's assigned event time timestamp
            current.lastModified = ctx.timestamp();

            // write the state back
            state.update(current);

            // schedule the next timer 60 seconds from the current event time
            ctx.timerService().registerEventTimeTimer(current.lastModified + 60000);


            out.collect(new Tuple2<>());
        }

        @Override
        public void onTimer(long timestamp, ProcessFunction<Tuple2<String, String>, Tuple2<String, Long>>.OnTimerContext ctx, Collector<Tuple2<String, Long>> out) throws Exception {
            // get the state for the key that scheduled the timer
            CountWithTimestamp result = state.value();

            // check if this is an outdated timer or the latest timer
            if (timestamp == result.lastModified + 60000) {
                // emit the state on timeout
                out.collect(new Tuple2<String, Long>(result.key, result.count));
            }
        }

        @Override
        public void open(OpenContext openContext) throws Exception {
            state = getRuntimeContext().getState(new ValueStateDescriptor<>("myState", CountWithTimestamp.class));
        }

        @Override
        public void close() throws Exception {
            super.close();
        }
    }

    public static class MyKeyedProcessFunction extends KeyedProcessFunction<Integer, Integer, String> {
        @Override
        public void processElement(Integer value, KeyedProcessFunction<Integer, Integer, String>.Context ctx, Collector<String> out) throws Exception {
            int taskIndex = getRuntimeContext().getTaskInfo().getIndexOfThisSubtask();
            ctx.getCurrentKey();

            ctx.timestamp();
            ctx.timerService();
            ctx.output(new OutputTag<>(""), "");

            out.collect("taskIndex=" + taskIndex + ",map-" + value);
        }
    }

    public static class CountWithTimestamp {

        public String key;
        public long count;
        public long lastModified;
    }

}
