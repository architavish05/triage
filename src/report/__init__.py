"""Reporting package re-export."""
from p4_app.report import *
from p4_app.report.build_report import (
    format_template_context,
    generate_situation_reports,
)
from p4_app.report.sms_summary import (
    ATTRIBUTION_SHORT,
    generate_sms_summary,
)

build = generate_situation_reports

__all__ = [
    "build",
    "format_template_context",
    "generate_situation_reports",
    "generate_sms_summary",
    "ATTRIBUTION_SHORT",
]
