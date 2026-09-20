"""Export only ANE activity fields; never publish the trace's process environment."""

import argparse
import hashlib
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from .common import save, stats


def summarize(hardware_xml, toc_xml):
    root = ET.parse(hardware_xml).getroot()
    identifiers = {e.attrib["id"]: e for e in root.iter() if "id" in e.attrib}

    def value(element):
        if "ref" in element.attrib:
            element = identifiers[element.attrib["ref"]]
        return element.attrib.get("fmt", element.text or "")

    def number(element):
        if "ref" in element.attrib:
            element = identifiers[element.attrib["ref"]]
        return int(element.text)

    events = []
    for row in root.findall(".//row"):
        events.append(
            {
                "start_ns": number(row.find("start-time")),
                "duration_ns": number(row.find("duration")),
                "device": value(row.find("ane-event-name")),
                "label": value(row.find("formatted-label")),
                "state": value(row.find("gpu-state")),
            }
        )
    toc = ET.parse(toc_xml).getroot()
    summary = toc.find(".//run/info/summary")
    return {
        "source_xml_sha256": hashlib.sha256(hardware_xml.read_bytes()).hexdigest(),
        "template": summary.findtext("template-name"),
        "instruments_version": summary.findtext("instruments-version"),
        "recording_seconds": float(summary.findtext("duration")),
        "scope": "ANE hardware intervals recorded during a separate CPU_AND_NE Laya workload; hardware table has no per-process or per-operator identity",
        "caveat": "Diagnostic tracing changes timing. These durations are not the uninstrumented end-to-end benchmark or an ANE power measurement. Core ML model-signpost table contained no rows in this trace.",
        "event_count": len(events),
        "devices": dict(Counter(e["device"] for e in events)),
        "labels": dict(Counter(e["label"] for e in events)),
        "states": dict(Counter(e["state"] for e in events)),
        "interval_duration": stats([e["duration_ns"] / 1e6 for e in events]) if events else None,
        "events": events,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware-xml", type=Path, required=True)
    parser.add_argument("--toc-xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.hardware_xml, args.toc_xml)
    save(args.output, result)
    print({key: value for key, value in result.items() if key != "events"})


if __name__ == "__main__":
    main()
