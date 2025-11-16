import os
import sys
import shutil

from collections import ChainMap, OrderedDict, defaultdict
from typing import Any
from torch import Tensor
from lightning_utilities.core.apply_func import apply_to_collection
from lightning.pytorch.loops.evaluation_loop import _EvaluationLoop
from lightning.pytorch.trainer.connectors.logger_connector.result import _OUT_DICT
from lightning.pytorch.callbacks.progress.rich_progress import _RICH_AVAILABLE


class SrEvalLoop(_EvaluationLoop):
    def on_run_end(self) -> list[dict[str, Tensor]]:
        """Runs the ``_on_evaluation_epoch_end`` hook."""
        # if `done` returned True before any iterations were done, this won't have been called in `on_advance_end`
        self.trainer._logger_connector.epoch_end_reached()
        self.trainer._logger_connector._evaluation_epoch_end()

        # hook
        self._on_evaluation_epoch_end()

        logged_outputs, self._logged_outputs = self._logged_outputs, []  # free memory
        # include any logged outputs on epoch_end
        epoch_end_logged_outputs = self.trainer._logger_connector.update_eval_epoch_metrics()
        all_logged_outputs = dict(ChainMap(*logged_outputs))  # list[dict] -> dict
        all_logged_outputs.update(epoch_end_logged_outputs)

        # for dl_outputs in logged_outputs:
            # dl_outputs.update(epoch_end_logged_outputs)
        
        logged_outputs.append(epoch_end_logged_outputs)

        # log metrics
        self.trainer._logger_connector.log_eval_end_metrics(all_logged_outputs)

        # hook
        self._on_evaluation_end()

        # enable train mode again
        self._on_evaluation_model_train()

        if self.verbose and self.trainer.is_global_zero:
            self._print_results(logged_outputs, self._stage.value)

        return logged_outputs

    @staticmethod
    def _print_results(results: list[dict[str, Tensor]], stage: str) -> None:
         # remove the dl idx suffix
        # results = [{k.split("/dataloader_idx_")[0]: v for k, v in result.items()} for result in results]
        headers = [list(result.keys())[0].split("/")[1] for result in results]
        results = [{k.split("/")[0]: v for k, v in result.items()} for result in results]
        metrics_paths = {k for keys in apply_to_collection(results, dict, _EvaluationLoop._get_keys) for k in keys}
        if not metrics_paths:
            return

        metrics_strs = [":".join(metric) for metric in metrics_paths]
        # sort both lists based on metrics_strs
        metrics_strs, metrics_paths = zip(*sorted(zip(metrics_strs, metrics_paths)))

        # headers = [list(result.keys())[0].split("/")[1] for result in results]

        # fallback is useful for testing of printed output
        term_size = shutil.get_terminal_size(fallback=(120, 30)).columns or 120
        max_length = int(min(max(len(max(metrics_strs, key=len)), len(max(headers, key=len)), 25), term_size / 2))

        rows: list[list[Any]] = [[] for _ in metrics_paths]

        for result in results:
            for metric, row in zip(metrics_paths, rows):
                val = _EvaluationLoop._find_value(result, metric)
                if val is not None:
                    if isinstance(val, Tensor):
                        val = val.item() if val.numel() == 1 else val.tolist()
                    row.append(f"{val}")
                else:
                    row.append(" ")

        # keep one column with max length for metrics
        num_cols = int((term_size - max_length) / max_length)

        for i in range(0, len(headers), num_cols):
            table_headers = headers[i : (i + num_cols)]
            table_rows = [row[i : (i + num_cols)] for row in rows]

            table_headers.insert(0, f"{stage} Metric".capitalize())

            if _RICH_AVAILABLE:
                from rich import get_console
                from rich.table import Column, Table

                columns = [Column(h, justify="center", style="magenta", width=max_length) for h in table_headers]
                columns[0].style = "cyan"

                table = Table(*columns)
                for metric, row in zip(metrics_strs, table_rows):
                    metric = metric.split("/")[0] # Addition. Kinda hacky
                    row.insert(0, metric)
                    table.add_row(*row)

                console = get_console()
                console.print(table)
            else:
                row_format = f"{{:^{max_length}}}" * len(table_headers)
                half_term_size = int(term_size / 2)

                try:
                    # some terminals do not support this character
                    if sys.stdout.encoding is not None:
                        "─".encode(sys.stdout.encoding)
                except UnicodeEncodeError:
                    bar_character = "-"
                else:
                    bar_character = "─"
                bar = bar_character * term_size

                lines = [bar, row_format.format(*table_headers).rstrip(), bar]
                for metric, row in zip(metrics_strs, table_rows):
                    # deal with column overflow
                    if len(metric) > half_term_size:
                        while len(metric) > half_term_size:
                            row_metric = metric[:half_term_size]
                            metric = metric[half_term_size:]
                            lines.append(row_format.format(row_metric, *row).rstrip())
                        lines.append(row_format.format(metric, " ").rstrip())
                    else:
                        lines.append(row_format.format(metric, *row).rstrip())
                lines.append(bar)
                print(os.linesep.join(lines))

