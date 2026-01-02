package flink.application.study.datastream.datasource.fileconnector;

import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.annotation.JsonProperty;
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.annotation.JsonPropertyOrder;

/**
 * 使用如下包的注解.
 * org.apache.flink.shaded.jackson2.com.fasterxml.jackson.annotation.*
 * <p>
 * 注解 @JsonPropertyOrder({"id", "name", "age"})
 * 与 @JsonProperty(index = 1)是等价的，两者使用其一即可.
 */

@JsonPropertyOrder({"id", "name", "age"})
public class User {

    @JsonProperty(index = 1)
    private String id;

    @JsonProperty(index = 2)
    private String name;

    @JsonProperty(index = 3)
    private String age;

    public User() {
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getAge() {
        return age;
    }

    public void setAge(String age) {
        this.age = age;
    }

    @Override
    public String toString() {
        return "User{" +
                "id='" + id + '\'' +
                ", name='" + name + '\'' +
                ", age='" + age + '\'' +
                '}';
    }
}
