from gpiozero import Servo
from time import sleep

servo1 = Servo(18)
servo2 = Servo(23)
delay = 0.02

try:
    while True:
        for pos in [i / 10.0 for i in range(-10, 11)]:
            servo1.value = pos
            servo2.value = -pos
            sleep(delay)
        for pos in [i / 10.0 for i in range(10, -11, -1)]:
            servo1.value = pos
            servo2.value = -pos
            sleep(delay)

except KeyboardInterrupt:
    print("\nReleasing GPIO...")
    servo1.detach()
    servo2.detach()
    servo1.close()
    servo2.close()
    print("Stopped safely.")

