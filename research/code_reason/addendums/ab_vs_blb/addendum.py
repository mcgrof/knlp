"""ab_vs_blb addendum: pair the paper-only (A) and augmented (blB) runs.

Unlike the analysis addendums, this one does not inspect the repository; it
records the intent to emit paired A (paper-only certificate) vs blB
(augmented certificate) artifacts for the same task so the two can be
compared directly. The actual pairing is written by ArtifactWriter.write_ab;
this envelope marks the addendum enabled/disabled from the flag so the run
config is self-describing.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import Addendum  # noqa: E402


class AbVsBlbAddendum(Addendum):
    name = "ab_vs_blb"
    flag = "CONFIG_CODE_REASON_ADDENDUM_A_VS_BLB"

    def run(self, task, reader, cert):
        return {
            "pairs": ["A.paper_only", "blB.augmented", "comparison"],
            "note": "pairing emitted by ArtifactWriter.write_ab",
        }
