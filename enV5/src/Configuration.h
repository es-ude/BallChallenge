#ifndef BALL_CHALLENGE_CONFIGURATION_H
#define BALL_CHALLENGE_CONFIGURATION_H

/* region MQTT */

#ifndef EIP_BASE
#define EIP_BASE "eip://uni-due.de/es"
#endif

#ifndef EIP_DEVICE_ID
#define EIP_DEVICE_ID "dataCollect01"
#endif

#ifndef TRIGGER_TOPIC
#define TRIGGER_TOPIC "MEASUREMENT"
#endif

#ifndef ACCELEROMETER_TOPIC
#define ACCELEROMETER_TOPIC "accelerometer"
#endif

#ifndef GYROSCOPE_TOPIC
#define GYROSCOPE_TOPIC "gyroscope"
#endif

#ifndef TIMER_TOPIC
#define TIMER_TOPIC "time"
#endif

/* endregion MQTT */

/* region SYSTEM */

#ifndef BATCH_INTERVALL
#define BATCH_INTERVALL (3)
#endif

#ifndef MEASUREMENT_FREQUENCY_ACCELEROMETER
#define MEASUREMENT_FREQUENCY_ACCELEROMETER (400)
#endif

#ifndef MEASUREMENT_FREQUENCY_GYROSCOPE
#define MEASUREMENT_FREQUENCY_GYROSCOPE (400)
#endif

/* endregion SYSTEM */

#endif // BALL_CHALLENGE_CONFIGURATION_H
