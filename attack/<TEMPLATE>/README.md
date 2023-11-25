
## Usage Examples

###### Extract features
```bash
python preprocess/extract.py --traces "${TRACE_PATH}" --output "${FEATURE_PATH}"
```

###### claaify

```bash
python analysis/info_leak.py --features "${FEATURE_PATH}" --output "${LEAKAGE_PATH}" \
                             --n_samples 5000 --nmi_threshold 0.9 --topn 100 --n_procs 8
```
