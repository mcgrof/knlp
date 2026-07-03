#!/usr/bin/env python3
"""Map line numbers across a unified diff (old <-> new).

Given the hunks of one file diff, translate a pre-patch line number to its
post-patch position and back, so fault-localization overlap and evidence
citations line up across patch versions. Pure arithmetic on hunk offsets.
"""

from __future__ import annotations


class LineMapper:
    def __init__(self, hunks):
        # hunks: list of objects with old_start/old_count/new_start/new_count
        self.hunks = sorted(hunks, key=lambda h: h.old_start)

    def old_to_new(self, old_line):
        offset = 0
        for h in self.hunks:
            old_end = h.old_start + h.old_count - 1
            if old_line < h.old_start:
                break
            if h.old_start <= old_line <= old_end:
                # inside a changed hunk: best-effort clamp to hunk's new start
                return None
            offset += h.new_count - h.old_count
        return old_line + offset

    def new_to_old(self, new_line):
        offset = 0
        for h in self.hunks:
            new_end = h.new_start + h.new_count - 1
            if new_line < h.new_start:
                break
            if h.new_start <= new_line <= new_end:
                return None
            offset += h.new_count - h.old_count
        return new_line - offset
