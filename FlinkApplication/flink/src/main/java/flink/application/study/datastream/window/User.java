package flink.application.study.datastream.window;

public class User {
    private String name;
    private Long timestamp;
    private Integer value;

    public User(String name, Long timestamp, Integer value) {
        this.name = name;
        this.timestamp = timestamp;
        this.value = value;
    }

    public String getName() {
        return name;
    }

    public Long getTimestamp() {
        return timestamp;
    }

    public Integer getValue() {
        return value;
    }

    @Override
    public String toString() {
        return "User{" + name +
                ", " + timestamp +
                ", " + value +
                '}';
    }
}
