package flink.application.study.datastream;

public class Word {
    private String word;
    private Integer count;
    private Long tt;

    /**
     * 必须有无参构造.
     */
    public Word() {
    }

    public Word(String word, Integer count, Long tt) {
        this.word = word;
        this.count = count;
        this.tt = tt;
    }

    public String getWord() {
        return word;
    }

    public void setWord(String word) {
        this.word = word;
    }

    public Long getTt() {
        return tt;
    }

    public void setTt(Long tt) {
        this.tt = tt;
    }

    public Integer getCount() {
        return count;
    }

    public void setCount(Integer count) {
        this.count = count;
    }

    @Override
    public String toString() {
        return "Word{" +
                "word='" + word + '\'' +
                ", count=" + count +
                ", tt=" + tt +
                '}';
    }

}
