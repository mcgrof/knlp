# SPDX-License-Identifier: MIT
"""Opt-in observations of the implemented trained carry matrices, without repair."""

import torch


class RoutingObservation:
    def __init__(self):
        self.values = []

    @torch.no_grad()
    def add(self, state, pre, post, carry):
        finite = torch.stack(
            [torch.isfinite(x).all() for x in (state, pre, post, carry)]
        ).all()
        values = torch.stack(
            (
                (carry.sum(-1) - 1).abs().amax(),
                (carry.sum(-2) - 1).abs().amax(),
                (-carry).clamp_min(0).amax(),
                torch.maximum(-pre, pre - 1).clamp_min(0).amax(),
                torch.maximum(-post, post - 2).clamp_min(0).amax(),
                (~finite).float(),
            )
        )
        self.values.append(
            torch.nan_to_num(values, nan=torch.finfo(torch.float32).max).detach()
        )

    def report(self, expected_calls, row_col_limit=1e-4, domain_limit=1e-7):
        if len(self.values) != expected_calls:
            raise RuntimeError("Incomplete trained-routing coverage")
        maxima = (
            torch.stack(self.values).amax(0).cpu().tolist()
            if self.values
            else [0.0] * 6
        )
        return dict(
            calls=len(self.values),
            row_sum_error=maxima[0],
            column_sum_error=maxima[1],
            negative_carry_error=maxima[2],
            pre_range_error=maxima[3],
            post_range_error=maxima[4],
            finite=maxima[5] == 0.0,
            row_col_limit=row_col_limit,
            domain_limit=domain_limit,
            constraint_pass=all(x <= row_col_limit for x in maxima[:2])
            and all(x <= domain_limit for x in maxima[2:5])
            and maxima[5] == 0.0,
            policy="Retain every violation; no routing repair, Sinkhorn/tolerance changes or seed omission",
        )
