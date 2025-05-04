package de.ude.ies.elastic_ai;

import de.ude.ies.elastic_ai.protocol.HivemqBroker;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.SocketException;
import java.net.UnknownHostException;
import org.springframework.context.annotation.Bean;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Component;

@Component
public class BallChallengeBeans {

    private final Environment env;

    public BallChallengeBeans(final Environment env) {
        this.env = env;
    }

    @Bean
    public String PublicIp() {
        String hostIp = env.getProperty("host.public_ip");
        if (hostIp == null) {
            try (final DatagramSocket socket = new DatagramSocket()) {
                socket.connect(InetAddress.getByName("8.8.8.8"), 10002);
                hostIp = socket.getLocalAddress().getHostAddress();
            } catch (UnknownHostException | SocketException e) {
                throw new RuntimeException(e);
            }
        }
        return hostIp.strip();
    }

    @Bean
    public Integer PublicPort() {
        return env.getProperty("server.port", Integer.class);
    }

    @Bean
    public Integer BrokerPort() {
        return env.getProperty("mqtt.broker.port", Integer.class);
    }

    @Bean
    public String BrokerIp() {
        return env.getProperty("mqtt.broker.ip", String.class);
    }

    @Bean
    public String BrokerBaseDomain() {
        String baseDomain = env.getProperty("mqtt.broker.base_domain", String.class);
        if (baseDomain == null) {
            baseDomain = "eip://uni-due.de/es";
        }
        return baseDomain;
    }

    @Bean
    public Integer CameraPort() {
        return env.getProperty("camera.port", Integer.class);
    }

    @Bean
    public String CameraIp() {
        return env.getProperty("camera.ip", String.class);
    }

    @Bean
    public BallChallengeEndpoint endpoint(
        String BrokerIp,
        Integer BrokerPort,
        String BrokerBaseDomain,
        String CameraIp,
        Integer CameraPort,
        String PublicIp,
        Integer PublicPort
    ) {
        BallChallengeEndpoint ballChallengeEndpoint = new BallChallengeEndpoint(
            PublicIp,
            PublicPort,
            CameraIp,
            CameraPort
        );
        ballChallengeEndpoint.bindToCommunicationEndpoint(
            new HivemqBroker(BrokerBaseDomain, BrokerIp, BrokerPort)
        );
        return ballChallengeEndpoint;
    }
}
