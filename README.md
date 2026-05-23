# Rule Twin - Industrial Control System Anomaly Detection System

Rule Twin is an industrial control system anomaly detection system based on physical rules and event correlation, supporting SWaT and WADI datasets.

## Quick Start

### SWaT Dataset

#### Original Attack Dataset
```bash
python sum.py ^
  --input SWaT_Dataset_Attack_v0.csv ^
  --output My_Results.csv ^
  --window_lit 90 ^
  --k 60 ^
  --trend_window 40 ^
  --backwash_window 60 ^
  --rules rules_summary.json ^
  --rules_act rules_merged.csv ^
  --rules_pair rules_summary_pairs.csv
```

#### Fixed Attack Dataset (Mainly addresses annotation issues in the original dataset)
```bash
python sum.py ^
  --input SWaT_Dataset_Attack_v0_fixed.csv ^
  --output My_Results_fixed.csv ^
  --window_lit 90 ^
  --k 60 ^
  --trend_window 40 ^
  --backwash_window 60 ^
  --rules rules_summary.json ^
  --rules_act rules_merged.csv ^
  --rules_pair rules_summary_pairs.csv
```

### WADI Dataset
```bash
python sumn.py ^
  --data_file "WADI_attackdataLABLE.csv" ^
  --rules_file "merged_tank_rules.csv" ^
  --output_file "WADI_Final_Fusion_Result.csv" ^
  --min_anomaly_length_model 60 ^
  --default_threshold 3.0 3.0 ^
  --trend_window 40 ^
  --fluctuation_val 0.5 ^
  --search_buffer 60 ^
  --threshold_pct 0.05 ^
  --max_gap_size_rule 10 ^
  --std_check_window 120
```

## Parameter Description

### Basic Input/Output Parameters

- `--input` (Default: SWaT_Dataset_Attack_v0.csv)
  - Path to the input dataset file. Supports .csv or .xlsx formats.

- `--output` (Default: SWaT_Integrated_Detection_Result.csv)
  - Path for the detection results output file.

- `--data_file` (WADI-specific)
  - Input file path for WADI dataset.

- `--rules_file` (WADI-specific)
  - Rules file path for WADI.

- `--output_file` (WADI-specific)
  - Output file path for WADI detection results.

### Physical Model Parameters (LIT/PIT/AIT)

- `--window_lit` (Default: 90)
  - Sliding window size for LIT physical model (in rows/seconds). Used for calculating fluid dynamics formulas and standard deviation.

### Event and Rule Detection Parameters (MX, BP, S1, S2, S3)

- `--k` (Default: 20)
  - Rule matching search radius. When a preceding event (e.g., low water level) is detected, search for the subsequent event (e.g., pump activation) within k rows before and after.

- `--trend_window` (Default: 10)
  - Sensor trend window. Used to determine if a sensor is continuously rising/falling and reaching thresholds.

- `--backwash_window` (Default: 60)
  - Backwash context window. Used to detect if the system is currently in backwash state (MV303=2) to suppress false positives for specific rules.

- `--min_anomaly_length_model` (WADI-specific, Default: 60)
  - Minimum anomaly length model parameter

- `--default_threshold` (WADI-specific, Default: 3.0 3.0)
  - Default threshold parameters

- `--fluctuation_val` (WADI-specific, Default: 0.5)
  - Fluctuation value parameter

- `--search_buffer` (WADI-specific, Default: 60)
  - Search buffer parameter

- `--threshold_pct` (WADI-specific, Default: 0.05)
  - Threshold percentage parameter

- `--max_gap_size_rule` (WADI-specific, Default: 10)
  - Maximum gap size rule parameter

- `--std_check_window` (WADI-specific, Default: 120)
  - Standard deviation check window parameter

### Rule Configuration File Paths

- `--rules` (Default: rules_summary.json)
  - JSON file defining sensor upper and lower limits.

- `--rules_act` (Default: rules_merged.csv)
  - CSV file defining "actuator->actuator"联动 rules.

- `--rules_pair` (Default: rules_summary_pairs.csv)
  - CSV file defining "sensor->actuator" pairing rules.


## Experimental Results

<img width="1369" height="529" alt="1a45b556e20f66eb00eb8b6d913cf299" src="https://github.com/user-attachments/assets/42801a94-6c3d-4d8c-bc98-2dc663ece506" />
<img width="1154" height="1074" alt="82661cb924a10cc246acdb1dc02050fa" src="https://github.com/user-attachments/assets/bb726ada-5f90-435c-81be-0be0e6dfe140" />



## System Architecture

The Rule Twin system consists of the following core components:

1. **Physical Model Engine (LIT/PIT/AIT)**
   - Physics-based anomaly detection
   - Sliding window calculations and statistical analysis

2. **Event Correlation Engine**
   - Sensor-actuator correlation rules
   - Time series pattern matching

3. **Rule Management System**
   - Rule configuration and loading
   - Dynamic rule updates

4. **Result Fusion Module**
   - Multi-source detection result fusion
   - Anomaly scoring and ranking

## Supported Datasets

- **SWaT (Secure Water Treatment)**
  - Water treatment system dataset
  - Contains normal and attack scenarios

- **WADI (Water Distribution)**
  - Water distribution system dataset
  - Large-scale sensor network data

## Output Format

Detection results are output in CSV format, containing the following columns:
- Timestamp
- Sensor ID
- Detected anomaly type
- Confidence score
- Rule matching information

## Dependencies

- Python 3.7+
- Pandas
- NumPy
- Scikit-learn
- Matplotlib (optional, for visualization)

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contribution

Contributions are welcome! Please submit Issues and Pull Requests to improve this project.


## References

**RuleTwin: Physics-Constrained Rule Induction for Anomaly Detection in Industrial Multivariate TimeSeries**

> Jingzheng Mao et al.  
> KDD '26: Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2  
> August 09--13, 2026, Jeju Island, Republic of Korea  
> DOI: [10.1145/3770855.3818020](https://doi.org/10.1145/3770855.3818020)

```bibtex
@inproceedings{rule_twin_2026,
  title={RuleTwin: Physics-Constrained Rule Induction for Anomaly Detection in Industrial Multivariate TimeSeries},
  author={Jingzheng Mao and Co-authors},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
  year={2026},
  pages={1-10},
  doi={10.1145/3770855.3818020},
  isbn={979-8-4007-2259-2/2026/08}
}
```
