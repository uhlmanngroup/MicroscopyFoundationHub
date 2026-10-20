# Results currently absent or clipped in the Results Report PDF

All figures are foreground IoU, mean ± sample standard deviation. Except where
marked, each entry uses five evaluated runs. These values were read from the
cluster's saved `metrics.json` and `config_used.yaml` files on 20 October 2026.
The current report is `1435df1c-df0c-48d6-9f98-d3102d73b1e8_Results_Report.pdf`.

## Individual EM: columns absent from the PDF

| Dataset | DINOv2 Frozen | DINOv2 LoRA | DINOv2 FT | DINOv3 LoRA | OpenCLIP LoRA |
|---|---:|---:|---:|---:|---:|
| VNC | 0.723 ± 0.002 | 0.818 ± 0.002 (3 runs; rerun pending) | 0.750 ± 0.023 | 0.875 ± 0.001 | 0.679 ± 0.017 |
| Lucchi++ | 0.717 ± 0.006 | 0.802 ± 0.007 | 0.742 ± 0.010 | 0.838 ± 0.002 | 0.624 ± 0.007 |
| Kasthuri++ | 0.744 ± 0.001 | 0.847 ± 0.001 | 0.746 ± 0.039 | 0.854 ± 0.001 | 0.788 ± 0.004 |

The older DINOv2 and DINOv3 individual EM configs do not record seeds, even
where five runs exist. Treat those estimates as existing run summaries, not
verified five-independent-seed results. The VNC DINOv2 LoRA value has only
three evaluated DINOv2-L runs and should be replaced by the new five-seed run.

## Individual DeepBacs: columns absent from the PDF

| Dataset | DINOv2 Frozen | DINOv2 LoRA | DINOv2 FT | DINOv3 LoRA |
|---|---:|---:|---:|---:|
| S. aureus | 0.626 ± 0.008 | 0.692 ± 0.011 | 0.802 ± 0.050 | 0.635 ± 0.007 |
| E. coli | 0.756 ± 0.003 | 0.842 ± 0.001 | 0.874 ± 0.002 | 0.804 ± 0.002 |
| B. subtilis | 0.566 ± 0.010 | 0.693 ± 0.007 | 0.762 ± 0.007 | 0.696 ± 0.005 |

These DeepBacs results use training seeds 1–5 and split seeds 101–105.

## Joint DeepBacs: DINOv3 LoRA column absent from the PDF

| Dataset | DINOv3 LoRA |
|---|---:|
| S. aureus | 0.729 ± 0.008 |
| E. coli | 0.817 ± 0.004 |
| B. subtilis | 0.731 ± 0.012 |
| Pooled foreground IoU | 0.777 ± 0.006 |

The PDF has the S. aureus and E. coli rows reversed for OpenCLIP's joint
DeepBacs Frozen and FT columns. The saved per-species metrics are:

| Dataset | OpenCLIP Frozen | OpenCLIP FT |
|---|---:|---:|
| S. aureus | 0.679 ± 0.008 | 0.816 ± 0.004 |
| E. coli | 0.735 ± 0.005 | 0.861 ± 0.003 |

The mapping is verified against the test image filename stems: `jenilered`
denotes S. aureus, `pos` denotes E. coli, and `test` denotes B. subtilis.

## Corrected joint EM: OpenCLIP columns clipped in the PDF

| Dataset | OpenCLIP Frozen | OpenCLIP LoRA | OpenCLIP FT |
|---|---:|---:|---:|
| VNC | 0.620 ± 0.008 | 0.780 ± 0.009 | 0.627 ± 0.069 |
| Kasthuri++ | 0.708 ± 0.009 | 0.820 ± 0.006 | 0.817 ± 0.013 |
| Lucchi++ | 0.622 ± 0.011 | 0.680 ± 0.012 | 0.587 ± 0.046 |
| Mean of three datasets | 0.650 ± 0.005 | 0.760 ± 0.005 | 0.677 ± 0.035 |

The PDF also has a DINOv2 FT VNC standard-deviation typo: use
`0.792 ± 0.030`, not `0.792 ± 0.30`.

## Still awaiting training

- DINOv2 DeepBacs joint triplet: Frozen, LoRA, and FT.
- OpenCLIP DeepBacs LoRA: S. aureus, E. coli, B. subtilis, and joint triplet.
- DINOv2-L VNC LoRA: fresh five-seed set to replace the three-run estimate.

The submission and automatic five-seed summary scripts are in this directory.
