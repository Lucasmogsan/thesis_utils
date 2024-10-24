
# nvidia_smi_logger.sh
Records the nvidia-smi specified parameters and logs it to a csv file.

```bash
cd nvidia_smi_logger
chmod +x nvidia_smi_logger.sh

./nvidia_smi_logger.sh <frequency> <output_directory>
```

# download_benchmar_data
Downloads the benchmarking dataset (TUM and Replica) and saves it locally (added to gitignore)

```bash
cd download_benchmark_data
chmod +x download_replica.sh
chmod +x download_tum.sh

./download_replica.sh
./download_tum.sh
```