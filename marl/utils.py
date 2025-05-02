import pandas as pd
import xml.etree.ElementTree as et
import os


def get_average_travel_time():
    path = os.path.join(os.path.dirname(__file__), '..', 'scenario', 'sample.tripinfo.xml')
    xtree = et.parse(path)
    xroot = xtree.getroot()
    rows = []
    for node in xroot:
        travel_time = node.attrib.get("duration")
        rows.append({"travel_time": travel_time})

    columns = ["travel_time"]
    travel_time = pd.DataFrame(rows, columns=columns).astype("float64")
    return travel_time["travel_time"].mean()
