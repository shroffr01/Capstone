from pathlib import Path
import xml.etree.ElementTree as et
import pandas as pd

def get_average_travel_time():
    xml_path = (
        Path(__file__)
        .resolve()
        .parent      # <repo>/marl
        .parent      # <repo>/
        / "scenario"
        / "sample.tripinfo.xml"
    )
    tree = et.parse(xml_path)
    root = tree.getroot()

    rows = [{"travel_time": float(node.attrib["duration"])} for node in root]
    return pd.DataFrame(rows).travel_time.mean()
