import random
import time
import json

# Generate messages from temperature sensor
# the temperature value
def get_temp_sensor_msg():
    val = str(round(random.normalvariate(25, 10), 1))
    valString = str(val)
    sensor = {}
    sensor['id'] = 1
    sensor["seqno"] = 1
    for i in range(10000):
        valString += valString
    valString += " C"
    msg = {
        "dev_id": str(sensor['id']),
        "ts": round(time.time(), 5),
        "seq_no": sensor["seqno"],
        "data_size": len(valString),
        "sensor_data": str(valString)
    }
    sensor["seqno"] += 1
    st = sensor["interval"]
    return json.dumps(msg), st

get_temp_sensor_msg()