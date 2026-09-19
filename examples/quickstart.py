"""Run after exporting models/laya-multilingual."""

import json

from laya_coreml import load

agent = load("models/laya-multilingual", compute_units="cpu_gpu")
result = agent.predict(
    "I was billed twice. Please refund the duplicate.",
    {
        "department": {
            "type": "choice",
            "instructions": "Who should handle this?",
            "criteria": ["billing", "technical", "sales"],
        }
    },
)
print(json.dumps(result, indent=2))
