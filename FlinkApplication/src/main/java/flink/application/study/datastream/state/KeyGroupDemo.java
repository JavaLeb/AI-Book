package flink.application.study.datastream.state;

import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.api.common.state.MapState;
import org.apache.flink.api.common.state.MapStateDescriptor;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.KeyedStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.util.Collector;
import org.apache.flink.util.MathUtils;

import java.util.Map;


public class KeyGroupDemo {
    public static void main(String[] args) throws Exception {

        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        int parallelism = 2;
        env.setParallelism(parallelism);
        env.setMaxParallelism(2);
        // 从数据源读取数据.
        DataStream<String> source1 = env.fromData("Hadoop", "MapReduce", "Tez");
        DataStream<String> source2 = env.fromData("ZooKeeper", "API", "Hadoop");
        // 标记数据源.
        DataStream<Word> wordSource1 = source1.map(element -> new Word(element, 1));
        DataStream<Word> wordSource2 = source2.map(element -> new Word(element, 2));
        // 合并数据源.
        DataStream<Word> source = wordSource1.union(wordSource2);
        // 指定Key.
        KeyedStream<Word, Integer> keyedStream = source.keyBy(new KeySelector<Word, Integer>() {
            @Override
            public Integer getKey(Word word) throws Exception {
                Integer key = word.getLen();
                word.setKey(key);

                return key;
            }
        });
        keyedStream.process(new MyKeyedProcessFunction()).forward().print("|--->");
        env.execute();
    }

    public static class MyKeyedProcessFunction extends KeyedProcessFunction<Integer, Word, Word> {
        private transient MapState<String, Integer> mapState;

        @Override
        public void processElement(Word word,
                                   KeyedProcessFunction<Integer, Word, Word>.Context ctx,
                                   Collector<Word> out) throws Exception {
            Integer count = mapState.get(word.getContent());
            count = (count == null ? 1 : count + 1);
            mapState.put(word.getContent(), count);
            System.out.println("|+++更新状态+++>word=" + word.getContent() + ",count=" + count);

            int taskIndex = getRuntimeContext().getTaskInfo().getIndexOfThisSubtask();
            int parallelism = getRuntimeContext().getTaskInfo().getNumberOfParallelSubtasks();
            int maxParallelism = getRuntimeContext().getTaskInfo().getMaxNumberOfParallelSubtasks();
            int hashCode = word.getKey().hashCode();
            int murmurHash = MathUtils.murmurHash(hashCode);
            int customKeyGroupId = murmurHash % maxParallelism;
            int customTaskIndex = customKeyGroupId * parallelism / maxParallelism;
            word.setHashCode(hashCode);
            word.setMurmurHash(murmurHash);
            word.setTaskIndex(taskIndex);
            word.setMaxParallelism(maxParallelism);
            word.setCustomKeyGroupId(customKeyGroupId);
            word.setCustomTaskIndex(customTaskIndex);

            out.collect(word);
        }

        @Override
        public void open(OpenContext openContext) throws Exception {
            super.open(openContext);
            // 初始化值状态.
            MapStateDescriptor<String, Integer> descriptor = new MapStateDescriptor<>(
                    "descriptor",
                    String.class,
                    Integer.class);

            this.mapState = getRuntimeContext().getMapState(descriptor);
        }

        @Override
        public void close() throws Exception {
            for (Map.Entry<String, Integer> entry : mapState.entries()) {
                int taskIndex = getRuntimeContext().getTaskInfo().getIndexOfThisSubtask();
                System.out.println("|--->taskIndex=" + taskIndex + ",(word=" + entry.getKey() + ",count=" + entry.getValue() + ")");
            }
            super.close();
        }
    }


    public static class Word {
        private String content;
        private Integer sourceId;
        private Integer key;
        private Integer hashCode;
        private Integer murmurHash;
        private Integer partitionIndex;
        private Integer taskIndex;
        private Integer customTaskIndex;
        private Integer customKeyGroupId;
        private Integer maxParallelism;

        public Word(String content, Integer sourceId) {
            this.content = content;
            this.sourceId = sourceId;
        }

        public Integer getKey() {
            return key;
        }

        public void setKey(Integer key) {
            this.key = key;
        }

        public Integer getLen() {
            if (content == null) {
                return 0;
            }
            return content.length();
        }

        public String getContent() {
            return content;
        }

        public void setContent(String content) {
            this.content = content;
        }

        public Integer getSourceId() {
            return sourceId;
        }

        public void setSourceId(Integer sourceId) {
            this.sourceId = sourceId;
        }

        public Integer getHashCode() {
            return hashCode;
        }

        public void setHashCode(Integer hashCode) {
            this.hashCode = hashCode;
        }

        public Integer getMurmurHash() {
            return murmurHash;
        }

        public void setMurmurHash(Integer murmurHash) {
            this.murmurHash = murmurHash;
        }

        public Integer getPartitionIndex() {
            return partitionIndex;
        }

        public void setPartitionIndex(Integer partitionIndex) {
            this.partitionIndex = partitionIndex;
        }

        public Integer getTaskIndex() {
            return taskIndex;
        }

        public void setTaskIndex(Integer taskIndex) {
            this.taskIndex = taskIndex;
        }

        public Integer getCustomTaskIndex() {
            return customTaskIndex;
        }

        public void setCustomTaskIndex(Integer customTaskIndex) {
            this.customTaskIndex = customTaskIndex;
        }

        public Integer getCustomKeyGroupId() {
            return customKeyGroupId;
        }

        public void setCustomKeyGroupId(Integer customKeyGroupId) {
            this.customKeyGroupId = customKeyGroupId;
        }

        public Integer getMaxParallelism() {
            return maxParallelism;
        }

        public void setMaxParallelism(Integer maxParallelism) {
            this.maxParallelism = maxParallelism;
        }

        @Override
        public String toString() {
            return "Word{" +
                    "content='" + content + '\'' +
                    ", sourceId=" + sourceId +
                    ", hashCode=" + hashCode +
                    ", murmurHash=" + murmurHash +
                    ", taskIndex=" + taskIndex +
                    ", customTaskIndex=" + customTaskIndex +
                    ", customKeyGroupId=" + customKeyGroupId +
                    ", maxParallelism=" + maxParallelism +
                    '}';
        }
    }
}
