import os
from resource import getrusage
import resource
import subprocess
import csv
from threading import Thread
from time import sleep, time
from typing import Any, Dict

import psutil


def get_results(fr: int):
    my_env = os.environ.copy()
    my_env["FR"] = str(fr)
    start_time = time()
    proc = subprocess.Popen(["./build/test"], env=my_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    result: Dict[str, Any] = {"frame_rate": fr}
    ps = psutil.Process(proc.pid)

    def watch():
        max_ram = 0
        while proc.poll() is None:
            max_ram = max(max_ram, ps.memory_info().rss)
            sleep(0.1)
        result["max_ram"] = max_ram

    Thread(target=watch).start()

    while True:
        line = proc.stdout.readline()
        if not line:
            break
        if line.startswith("CAPTURE_"):
            result[line[8: line.index("=")].lower()] = int(line[line.index("=") + 1:])
    end_time = time()
    t = ps.cpu_times()
    result['cpu'] = (t.system + t.user) * 100 / (end_time - start_time)

    while proc.poll() is None:
        sleep(0.1)
    sleep(0.5)
    print(f"{fr=}, {result}")
    return result


with open("perf_results.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["frame_rate", "rendered", "dropped", "max_ram", "cpu"])
    writer.writeheader()
    for fr in range(30, 180, 5):
        results = get_results(fr)
        writer.writerow(results)
        # break
