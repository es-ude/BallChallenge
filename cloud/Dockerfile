FROM openjdk:17-alpine

RUN addgroup ballchallenge && \
  adduser --disabled-password --ingroup ballchallenge ballchallenge

WORKDIR /usr/src/ballchallenge
COPY app/build/libs/ballchallenge.jar ballchallenge.jar

USER ballchallenge

EXPOSE 80

CMD ["java", "-jar", "ballchallenge.jar"]
