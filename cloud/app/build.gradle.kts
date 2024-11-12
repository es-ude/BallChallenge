plugins {
    id("java")
    id("application")

    id("jacoco")
    id("nebula.facet") version "9.6.3"
    id("com.adarshr.test-logger") version "4.0.0"

    id("org.springframework.boot") version "3.3.5"
    id("io.spring.dependency-management") version "1.1.6"

    id("com.google.cloud.tools.jib") version "3.4.4"
}

repositories {
    mavenCentral()

    maven {
        url = uri("https://maven.pkg.github.com/es-ude/elastic.ai-runtime.cloud")
    }
}

dependencies {
    // region TEST dependencies
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")

    testImplementation(libs.junit.jupiter)
    testImplementation("org.testcontainers:testcontainers:1.20.3")
    testImplementation("org.testcontainers:junit-jupiter:1.20.3")
    testImplementation("org.testcontainers:hivemq:1.20.3")
    testImplementation("org.slf4j:slf4j-api:2.0.32")
    testImplementation("ch.qos.logback:logback-core:1.4.14")
    testImplementation("ch.qos.logback:logback-classic:1.4.12")
    testImplementation("org.springframework.boot:spring-boot-starter-test")
    // endregion TEST dependencies

    // region IMPLEMENTATION dependencies
    implementation("de.ude.ies.elastic_ai.cloud:runtime:0.0.1")

    implementation(libs.guava)
    implementation("com.google.guava:guava:33.0.0-jre")

    implementation("org.springframework.boot:spring-boot-starter-data-rest")
    implementation("org.springframework.boot:spring-boot-starter-thymeleaf")
    implementation("org.springframework.boot:spring-boot-starter-web")
    implementation("org.springframework.boot:spring-boot-starter-web-services")

    implementation("org.webjars:jquery:3.6.3")
    implementation("org.webjars:bootstrap:5.2.3")
    implementation("org.webjars:font-awesome:6.2.0")

    implementation("org.springframework.session:spring-session-core")
    implementation("net.sourceforge.argparse4j:argparse4j:0.9.0")

    compileOnly("org.projectlombok:lombok:1.18.30")

    annotationProcessor("org.springframework.boot:spring-boot-configuration-processor")
    annotationProcessor("org.projectlombok:lombok")
    // endregion IMPLEMENTATION dependencies
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(22)
    }
}

application {
    group = "de.ude.ies.elastic_ai.ball_challenge"
    mainClass = "de.ude.ies.elastic_ai.BallChallengeApplication"
}

tasks.named<Test>("test") {
    // Use JUnit Platform for unit tests.
    useJUnitPlatform()
    testLogging {
        events("passed", "skipped", "failed")
    }
    finalizedBy("jacocoTestReport")
}

tasks.jacocoTestReport {
    dependsOn(tasks.test)

    reports {
        xml.required.set(false)
        csv.required.set(false)
        html.outputLocation.set(layout.projectDirectory.dir("jacocoReports"))
    }
}

tasks.clean {
    doFirst {
        val jacocoReports = File("${projectDir}/jacocoReport")
        jacocoReports.delete()
    }
}
