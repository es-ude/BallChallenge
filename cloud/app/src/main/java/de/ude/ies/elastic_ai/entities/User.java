package de.ude.ies.elastic_ai.entities;

import jakarta.persistence.*;
import java.util.UUID;
import lombok.Getter;
import lombok.Setter;

@Entity
@Getter
@Table(name = "participants")
public class User {

    public enum Hand {
        RIGHT,
        LEFT,
    }

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    @Column(unique = true, nullable = false, updatable = false)
    private UUID id;

    @Setter
    @Column(unique = true, nullable = false, name = "uname")
    private String name;

    @Setter
    @Column(nullable = false)
    private float armLength;

    @Setter
    @Column(nullable = false)
    private float shoulderHeight;

    @Setter
    @Column(nullable = false)
    private Hand dominantHand;

    public User() {}

    public User(String name, float armLength, float shoulderHeight, Hand dominantHand) {
        this.name = name.strip();
        this.armLength = armLength;
        this.shoulderHeight = shoulderHeight;
        this.dominantHand = dominantHand;
    }

    @Override
    public String toString() {
        return (
            "User{" +
            "uuid=" +
            id +
            ", name='" +
            name +
            '\'' +
            ", armLength=" +
            armLength +
            ", shoulderHeight=" +
            shoulderHeight +
            ", dominantHand=" +
            dominantHand +
            '}'
        );
    }
}
