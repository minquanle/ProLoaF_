# Data provenance

The reproduction does not duplicate large raw OPSD CSV files in this repository.
After a recursive clone, the paper source's nested `data` submodule provides the
prepared file used by the original configuration:

```text
paper_source/data/opsd.csv
```

It has 43,824 hourly rows, covers 2015-01-01 to 2019-12-31, and contains the German
load target, Austrian load, weather variables, and calendar features needed by the
paper configuration. The source preprocessing configuration also documents the raw
OPSD Time Series release `2020-10-06` and weather-data release `2020-09-16`.

Local raw downloads may be kept in `data/raw/`; they are intentionally ignored by Git.
